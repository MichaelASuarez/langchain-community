from __future__ import annotations

import atexit
import csv
import enum
import json
import logging
import uuid
from contextlib import contextmanager
from io import StringIO
from threading import local
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.document import Document

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConnection
    from psycopg2.extensions import cursor as PgCursor


class Yellowbrick(VectorStore):
    """Yellowbrick as a vector database.
    Example:
        .. code-block:: python
            from langchain_community.vectorstores import Yellowbrick
            from langchain_community.embeddings.openai import OpenAIEmbeddings
            ...
    """

    class IndexType(str, enum.Enum):
        """Enumerator for the supported Index types within Yellowbrick."""

        NONE = "none"
        LSH = "lsh"
        IVF = "ivf"

    class IndexParams:
        """Parameters for configuring a Yellowbrick index."""

        def __init__(
            self,
            index_type: Optional["Yellowbrick.IndexType"] = None,
            params: Optional[Dict[str, Any]] = None,
        ):
            if index_type is None:
                index_type = Yellowbrick.IndexType.NONE
            self.index_type = index_type
            self.params = params or {}

        def get_param(self, key: str, default: Any = None) -> Any:
            return self.params.get(key, default)

    def __init__(
        self,
        embedding: Embeddings,
        connection_string: str,
        table: str,
        *,
        schema: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        drop: bool = False,
    ) -> None:
        """Initialize with yellowbrick client.
        Args:
            embedding: Embedding operator
            connection_string: Format 'postgres://username:password@host:port/database'
            table: Table used to store / retrieve embeddings from
        """
        from psycopg2 import extras

        extras.register_uuid()

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.ERROR)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        if not isinstance(embedding, Embeddings):
            self.logger.error("embeddings input must be Embeddings object.")
            return

        self.LSH_INDEX_TABLE: str = "_lsh_index"
        self.LSH_HYPERPLANE_TABLE: str = "_lsh_hyperplane"
        self.IVF_INDEX_TABLE: str = "_ivf_index"
        self.IVF_CENTROID_TABLE: str = "_ivf_centroid"
        self.CONTENT_TABLE: str = "_content"

        self.connection_string = connection_string
        self.connection = Yellowbrick.DatabaseConnection(connection_string, self.logger)
        atexit.register(self.connection.close_connection)

        self._schema = schema
        self._table = table
        self._embedding = embedding
        self._max_embedding_len = None
        self._check_database_utf8()

        with self.connection.get_cursor() as cursor:
            if drop:
                self.drop(table=self._table, schema=self._schema, cursor=cursor)
                self.drop(
                    table=self._table + self.CONTENT_TABLE,
                    schema=self._schema,
                    cursor=cursor,
                )
                self._drop_lsh_index_tables(cursor)

            self._create_schema(cursor)
            self._create_table(cursor)

    class DatabaseConnection:
        _instance = None
        _connection_string: str
        _thread_local = local()  # Thread-local storage for connections
        _logger: logging.Logger

        def __new__(
            cls, connection_string: str, logger: logging.Logger
        ) -> "Yellowbrick.DatabaseConnection":
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._connection_string = connection_string
                cls._instance._logger = logger
            return cls._instance

        def close_connection(self) -> None:
            connection = getattr(self._thread_local, "connection", None)
            if connection and not connection.closed:
                connection.close()
                self._thread_local.connection = None

        def get_connection(self) -> "PgConnection":
            import psycopg2

            connection = getattr(self._thread_local, "connection", None)
            try:
                if not connection or connection.closed:
                    connection = psycopg2.connect(self._connection_string)
                    connection.autocommit = False
                    self._thread_local.connection = connection
                else:
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
            except (Exception, psycopg2.DatabaseError) as error:
                self._logger.error(
                    f"Error detected, reconnecting: {error}", exc_info=False
                )
                connection = psycopg2.connect(self._connection_string)
                connection.autocommit = False
                self._thread_local.connection = connection

            return connection

        @contextmanager
        def get_managed_connection(self) -> Generator["PgConnection", None, None]:
            from psycopg2 import DatabaseError

            conn = self.get_connection()
            try:
                yield conn
            except DatabaseError as e:
                conn.rollback()
                self._logger.error(
                    "Database error occurred, rolling back transaction.", exc_info=True
                )
                raise RuntimeError("Database transaction failed.") from e
            else:
                conn.commit()

        @contextmanager
        def get_cursor(self) -> Generator["PgCursor", None, None]:
            with self.get_managed_connection() as conn:
                cursor = conn.cursor()
                try:
                    yield cursor
                finally:
                    cursor.close()

    def _create_schema(self, cursor: "PgCursor") -> None:
        """
        Helper function: create schema if not exists
        """
        from psycopg2 import sql

        if self._schema:
            cursor.execute(
                sql.SQL(
                    """
                    CREATE SCHEMA IF NOT EXISTS {s}
                """
                ).format(
                    s=sql.Identifier(self._schema),
                )
            )

    def _create_table(self, cursor: "PgCursor") -> None:
        """
        Helper function: create table if not exists
        """
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        t = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        c = sql.Identifier(self._table + self.CONTENT_TABLE + "_pk_doc_id")
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {t} (
                doc_id UUID NOT NULL,
                text VARCHAR(60000) NOT NULL,
                metadata JSONB NOT NULL,
                CONSTRAINT {c} PRIMARY KEY (doc_id))
                DISTRIBUTE ON (doc_id) SORT ON (doc_id)
            """
            ).format(
                t=t,
                c=c,
            )
        )

        schema_prefix = (self._schema,) if self._schema else ()
        t1 = sql.Identifier(*schema_prefix, self._table)
        t2 = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        c1 = sql.Identifier(
            self._table + self.CONTENT_TABLE + "_pk_doc_id_embedding_id"
        )
        c2 = sql.Identifier(self._table + self.CONTENT_TABLE + "_fk_doc_id")
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {t1} (
                doc_id UUID NOT NULL,
                embedding_id SMALLINT NOT NULL,
                embedding FLOAT NOT NULL,
                CONSTRAINT {c1} PRIMARY KEY (doc_id, embedding_id),
                CONSTRAINT {c2} FOREIGN KEY (doc_id) REFERENCES {t2}(doc_id))
                DISTRIBUTE ON (doc_id) SORT ON (doc_id)
            """
            ).format(
                t1=t1,
                t2=t2,
                c1=c1,
                c2=c2,
            )
        )

    def drop(
        self,
        table: str,
        schema: Optional[str] = None,
        cursor: Optional["PgCursor"] = None,
    ) -> None:
        """
        Helper function: Drop data. If a cursor is provided, use it;
        otherwise, obtain a new cursor for the operation.
        """
        if cursor is None:
            with self.connection.get_cursor() as cursor:
                self._drop_table(cursor, table, schema=schema)
        else:
            self._drop_table(cursor, table, schema=schema)

    def _drop_table(
        self,
        cursor: "PgCursor",
        table: str,
        schema: Optional[str] = None,
    ) -> None:
        """
        Executes the drop table command using the given cursor.
        """
        from psycopg2 import sql

        if schema:
            table_name = sql.Identifier(schema, table)
        else:
            table_name = sql.Identifier(table)

        drop_table_query = sql.SQL(
            """
        DROP TABLE IF EXISTS {} CASCADE
        """
        ).format(table_name)
        cursor.execute(drop_table_query)

    def _check_database_utf8(self) -> bool:
        """
        Helper function: Test the database is UTF-8 encoded
        """
        with self.connection.get_cursor() as cursor:
            query = """
                SELECT pg_encoding_to_char(encoding)
                FROM pg_database
                WHERE datname = current_database();
            """
            cursor.execute(query)
            encoding = cursor.fetchone()[0]

        if encoding.lower() == "utf8" or encoding.lower() == "utf-8":
            return True
        else:
            raise Exception("Database encoding is not UTF-8")

        return False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        batch_size = 10000

        texts = list(texts)
        embeddings = self._embedding.embed_documents(list(texts))
        results = []
        if not metadatas:
            metadatas = [{} for _ in texts]

        index_params = kwargs.get("index_params") or Yellowbrick.IndexParams()

        with self.connection.get_cursor() as cursor:
            content_io = StringIO()
            embeddings_io = StringIO()
            content_writer = csv.writer(
                content_io, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            embeddings_writer = csv.writer(
                embeddings_io, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            current_batch_size = 0

            for i, text in enumerate(texts):
                doc_uuid = str(uuid.uuid4())
                results.append(doc_uuid)

                content_writer.writerow([doc_uuid, text, json.dumps(metadatas[i])])

                for embedding_id, embedding in enumerate(embeddings[i]):
                    embeddings_writer.writerow([doc_uuid, embedding_id, embedding])

                current_batch_size += 1

                if current_batch_size >= batch_size:
                    self._copy_to_db(cursor, content_io, embeddings_io)

                    content_io.seek(0)
                    content_io.truncate(0)
                    embeddings_io.seek(0)
                    embeddings_io.truncate(0)
                    current_batch_size = 0

            if current_batch_size > 0:
                self._copy_to_db(cursor, content_io, embeddings_io)

        if index_params:
            self._update_index(index_params, uuid.UUID(doc_uuid))

        return results

    def _copy_to_db(
        self, cursor: "PgCursor", content_io: StringIO, embeddings_io: StringIO
    ) -> None:
        content_io.seek(0)
        embeddings_io.seek(0)

        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        table = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        content_copy_query = sql.SQL(
            """
            COPY {table} (doc_id, text, metadata) FROM 
            STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '\"')
        """
        ).format(table=table)
        cursor.copy_expert(content_copy_query, content_io)

        schema_prefix = (self._schema,) if self._schema else ()
        table = sql.Identifier(*schema_prefix, self._table)
        embeddings_copy_query = sql.SQL(
            """
            COPY {table} (doc_id, embedding_id, embedding) FROM 
            STDIN WITH (FORMAT CSV, DELIMITER E'\\t', QUOTE '\"')
        """
        ).format(table=table)
        cursor.copy_expert(embeddings_copy_query, embeddings_io)

    @classmethod
    def from_texts(
        cls: Type[Yellowbrick],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        connection_string: str = "",
        table: str = "langchain",
        schema: str = "public",
        drop: bool = False,
        **kwargs: Any,
    ) -> Yellowbrick:
        """Add texts to the vectorstore index.
        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            connection_string: URI to Yellowbrick instance
            embedding: Embedding function
            table: table to store embeddings
            kwargs: vectorstore specific parameters
        """
        vss = cls(
            embedding=embedding,
            connection_string=connection_string,
            table=table,
            schema=schema,
            drop=drop,
        )
        vss.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return vss

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by uuids.

        Args:
            ids: List of ids to delete, where each id is a uuid string.
        """
        from psycopg2 import sql

        if delete_all:
            where_sql = sql.SQL(
                """
                WHERE 1=1
            """
            )
        elif ids is not None:
            uuids = tuple(sql.Literal(id) for id in ids)
            ids_formatted = sql.SQL(", ").join(uuids)
            where_sql = sql.SQL(
                """
                WHERE doc_id IN ({ids})
            """
            ).format(
                ids=ids_formatted,
            )
        else:
            raise ValueError("Either ids or delete_all must be provided.")

        schema_prefix = (self._schema,) if self._schema else ()
        with self.connection.get_cursor() as cursor:
            table_identifier = sql.Identifier(
                *schema_prefix, self._table + self.CONTENT_TABLE
            )
            query = sql.SQL("DELETE FROM {table} {where_sql}").format(
                table=table_identifier, where_sql=where_sql
            )
            cursor.execute(query)

            table_identifier = sql.Identifier(*schema_prefix, self._table)
            query = sql.SQL("DELETE FROM {table} {where_sql}").format(
                table=table_identifier, where_sql=where_sql
            )
            cursor.execute(query)

            if self._table_exists(
                cursor, self._table + self.LSH_INDEX_TABLE, *schema_prefix
            ):
                table_identifier = sql.Identifier(
                    *schema_prefix, self._table + self.LSH_INDEX_TABLE
                )
                query = sql.SQL("DELETE FROM {table} {where_sql}").format(
                    table=table_identifier, where_sql=where_sql
                )
                cursor.execute(query)

        return None

    def _table_exists(
        self, cursor: "PgCursor", table_name: str, schema: str = "public"
    ) -> bool:
        """
        Checks if a table exists in the given schema
        """
        from psycopg2 import sql

        schema = sql.Literal(schema)
        table_name = sql.Literal(table_name)
        cursor.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM sys.table t INNER JOIN sys.schema s ON t.schema_id = s.schema_id
                WHERE s.name = {schema} AND t.name = {table_name}
            """
            ).format(
                schema=schema,
                table_name=table_name,
            )
        )
        return cursor.fetchone()[0] > 0

    def _generate_vector_uuid(self, vector: List[float]) -> uuid.UUID:
        import hashlib

        vector_str = ",".join(map(str, vector))
        hash_object = hashlib.sha1(vector_str.encode())
        hash_digest = hash_object.digest()
        vector_uuid = uuid.UUID(bytes=hash_digest[:16])
        return vector_uuid

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Yellowbrick with vector

        Args:
            embedding (List[float]): query embedding
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document, float]: List of Documents and scores
        """
        from psycopg2 import sql
        from psycopg2.extras import execute_values

        index_params = kwargs.get("index_params") or Yellowbrick.IndexParams()
        whereClause = "1=1"
        if kwargs.get("filter") is not None:
            filter_value = kwargs.get("filter")
            if filter_value is not None:
                jsFilter = json.loads(filter_value)
                whereClause = jsonFilter2sqlWhere(jsFilter, "v3.metadata")

        with self.connection.get_cursor() as cursor:
            tmp_embeddings_table = "tmp_" + self._table
            tmp_doc_id = self._generate_vector_uuid(embedding)
            create_table_query = sql.SQL(
                """ 
                CREATE TEMPORARY TABLE {} (
                doc_id UUID,
                embedding_id SMALLINT,
                embedding FLOAT)
                ON COMMIT DROP
                DISTRIBUTE REPLICATE
            """
            ).format(sql.Identifier(tmp_embeddings_table))
            cursor.execute(create_table_query)
            data_input = [
                (str(tmp_doc_id), embedding_id, embedding_value)
                for embedding_id, embedding_value in enumerate(embedding)
            ]
            insert_query = sql.SQL(
                "INSERT INTO {} (doc_id, embedding_id, embedding) VALUES %s"
            ).format(sql.Identifier(tmp_embeddings_table))
            execute_values(cursor, insert_query, data_input)

            schema_prefix = (self._schema,) if self._schema else ()
            embeddings_table = sql.Identifier(*schema_prefix, self._table)
            content_table = sql.Identifier(
                *schema_prefix, self._table + self.CONTENT_TABLE
            )
            if index_params.index_type == Yellowbrick.IndexType.LSH:
                tmp_hash_table = self._table + "_tmp_hash"
                self._generate_tmp_lsh_hashes(
                    cursor,
                    tmp_embeddings_table,
                    tmp_hash_table,
                )

                schema_prefix = (self._schema,) if self._schema else ()
                lsh_index = sql.Identifier(
                    *schema_prefix, self._table + self.LSH_INDEX_TABLE
                )
                input_hash_table = sql.Identifier(tmp_hash_table)
                sql_query = sql.SQL(
                    """
                    WITH index_docs AS (
                    SELECT
                        t1.doc_id,
                        SUM(ABS(t1.hash-t2.hash)) as hamming_distance
                    FROM
                        {lsh_index} t1
                    INNER JOIN
                        {input_hash_table} t2
                    ON t1.hash_index = t2.hash_index
                    GROUP BY t1.doc_id
                    HAVING hamming_distance <= {hamming_distance}
                    )
                    SELECT
                        text,
                        metadata,
                       SUM(v1.embedding * v2.embedding) /
                        (SQRT(SUM(v1.embedding * v1.embedding)) *
                       SQRT(SUM(v2.embedding * v2.embedding))) AS score
                    FROM
                        {tmp_embeddings_table} v1
                    INNER JOIN
                        {embeddings_table} v2
                    ON v1.embedding_id = v2.embedding_id
                    INNER JOIN
                        {content_table} v3
                    ON v2.doc_id = v3.doc_id
                    INNER JOIN
                        index_docs v4
                    ON v2.doc_id = v4.doc_id
                    where {whereClause}
                    GROUP BY v3.doc_id, v3.text, v3.metadata
                    ORDER BY score DESC
                    LIMIT %s
                """
                ).format(
                    tmp_embeddings_table=sql.Identifier(tmp_embeddings_table),
                    embeddings_table=embeddings_table,
                    content_table=content_table,
                    lsh_index=lsh_index,
                    input_hash_table=input_hash_table,
                    whereClause=sql.SQL(whereClause),
                    hamming_distance=sql.Literal(
                        index_params.get_param("hamming_distance", 0)
                    ),
                )
                cursor.execute(
                    sql_query,
                    (k,),
                )
                results = cursor.fetchall()
            elif index_params.index_type == Yellowbrick.IndexType.IVF:
                centroids_table = sql.Identifier(
                    *schema_prefix, self._table + self.IVF_CENTROID_TABLE
                )
                ivf_index_table = sql.Identifier(
                    *schema_prefix, self._table + self.IVF_INDEX_TABLE
                )
                quantization = index_params.get_param("quantization", False)

                tmp_quantized_embedding_sql = sql.SQL(
                    """
                     (SELECT doc_id, embedding_id, 
                      ROUND(((embedding + 1) / 2) * 255) AS embedding 
                     FROM {embedding_table})
                """
                ).format(
                    embedding_table=sql.Identifier(tmp_embeddings_table),
                )

                tmp_embedding_sql = sql.SQL(
                    """
                    (SELECT doc_id, embedding_id, embedding FROM {embedding_table})
                """
                ).format(
                    embedding_table=sql.Identifier(tmp_embeddings_table),
                )

                if quantization:
                    tmp_embedding = tmp_quantized_embedding_sql
                else:
                    tmp_embedding = tmp_embedding_sql

                '''
                centroid_id = sql.Literal(
                    self._find_centroid(cursor, "tmp_" + self._table)
                )
                sql_query = sql.SQL(
                    """
                    SELECT
                        text,
                        metadata,
                        score
                    FROM
                        (SELECT
                            v5.doc_id doc_id,
                            SUM(v1.embedding * v5.embedding) /
                            (SQRT(SUM(v1.embedding * v1.embedding)) *
                            SQRT(SUM(v5.embedding * v5.embedding))) AS score
                        FROM
                            {tmp_embedding} v1
                        INNER JOIN
                            {ivf_index_table} v5
                        ON v1.embedding_id = v5.embedding_id
                        WHERE v5.id = {centroid_id}
                        GROUP BY v5.doc_id
                        ORDER BY score DESC LIMIT %s
                        ) v4
                    INNER JOIN
                        {content_table} v3
                    ON v4.doc_id = v3.doc_id
                    where {whereClause}
                    ORDER BY score DESC
                """
                ).format(
                    content_table=content_table,
                    ivf_index_table=ivf_index_table,
                    centroid_id=centroid_id,
                    tmp_embedding=tmp_embedding,
                    whereClause=sql.SQL(whereClause)
                )
                cursor.execute(sql_query, (k,))
                results = cursor.fetchall()
                '''

                centroid_sql = sql.SQL(
                    """
                    WITH CentroidDistances AS (
                    SELECT
                        e.doc_id AS edoc_id,
                        c.id AS cdoc_id,
                        SUM(e.embedding * c.centroid) / 
                        (SQRT(SUM(c.centroid * c.centroid)) * 
                        SQRT(SUM(e.embedding * e.embedding))) AS cosine_similarity
                    FROM {tmp_embedding} e
                    JOIN {centroids_table} c ON e.embedding_id = c.centroid_id
                    GROUP BY edoc_id, cdoc_id   
                    ),
                    MaxSimilarities AS (
                    SELECT
                        edoc_id,
                        cdoc_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY edoc_id ORDER BY cosine_similarity DESC
                    ) AS rank
                    FROM CentroidDistances
                    ),
                    Centroid AS (
                    SELECT
                        cdoc_id
                        FROM MaxSimilarities
                    WHERE rank = 1
                    )
                """
                ).format(
                    centroids_table=centroids_table,
                    tmp_embedding=tmp_embedding_sql,
                )

                sql_query = sql.SQL(
                    """
                    {centroid_sql}
                    SELECT 
                        text,
                        metadata,
                        score
                    FROM
                        (SELECT
                            v5.doc_id doc_id,
                            SUM(v1.embedding * v5.embedding) /
                            (SQRT(SUM((v1.embedding * v1.embedding)::float)) *
                            SQRT(SUM((v5.embedding * v5.embedding)::float))) AS score
                        FROM
                            {tmp_embedding} v1
                        INNER JOIN
                            {ivf_index_table} v5
                        ON v1.embedding_id = v5.embedding_id
                        INNER JOIN
                            Centroid c
                        ON v5.id = c.cdoc_id
                        GROUP BY v5.doc_id
                        ORDER BY score DESC 
                        ) v4
                    INNER JOIN
                        {content_table} v3
                    ON v4.doc_id = v3.doc_id
                    where {whereClause}
                    ORDER BY score DESC
                    LIMIT %s
                """
                ).format(
                    centroid_sql=centroid_sql,
                    content_table=content_table,
                    ivf_index_table=ivf_index_table,
                    tmp_embedding=tmp_embedding,
                    whereClause=sql.SQL(whereClause),
                )
                cursor.execute(sql_query, (k,))
                results = cursor.fetchall()

            else:
                sql_query = sql.SQL(
                    """
                    SELECT 
                        text,
                        metadata,
                        score
                    FROM
                        (SELECT
                            v2.doc_id doc_id,
                            SUM(v1.embedding * v2.embedding) /
                            (SQRT(SUM(v1.embedding * v1.embedding)) *
                            SQRT(SUM(v2.embedding * v2.embedding))) AS score
                        FROM
                            {tmp_embeddings_table} v1
                        INNER JOIN
                            {embeddings_table} v2
                        ON v1.embedding_id = v2.embedding_id
                        GROUP BY v2.doc_id
                        ORDER BY score DESC
                        ) v4
                    INNER JOIN
                        {content_table} v3
                    ON v4.doc_id = v3.doc_id
                    where {whereClause}
                    ORDER BY score DESC
                    LIMIT %s
                """
                ).format(
                    tmp_embeddings_table=sql.Identifier(tmp_embeddings_table),
                    embeddings_table=embeddings_table,
                    content_table=content_table,
                    whereClause=sql.SQL(whereClause),
                )
                cursor.execute(sql_query, (k,))
                results = cursor.fetchall()

        documents: List[Tuple[Document, float]] = []
        for result in results:
            metadata = result[1] or {}
            doc = Document(page_content=result[0], metadata=metadata)
            documents.append((doc, result[2]))

        return documents

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with Yellowbrick

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of Documents
        """
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return [doc for doc, _ in documents]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Perform a similarity search with Yellowbrick

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of (Document, similarity)
        """
        embedding = self._embedding.embed_query(query)
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search with Yellowbrick by vectors

        Args:
            embedding (List[float]): query embedding
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

            NOTE: Please do not let end-user fill this and always be aware
                  of SQL injection.

        Returns:
            List[Document]: List of documents
        """
        documents = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, **kwargs
        )
        return [doc for doc, _ in documents]

    def migrate_schema_v1_to_v2(self) -> None:
        from psycopg2 import sql

        try:
            with self.connection.get_cursor() as cursor:
                schema_prefix = (self._schema,) if self._schema else ()
                embeddings = sql.Identifier(*schema_prefix, self._table)
                # For the RENAME TO statement the destination must be unqualified
                # (no schema).
                # Build both a schema-qualified identifier for later references and an
                # unqualified identifier to use as the target of the RENAME TO.
                old_embeddings_qualified = sql.Identifier(
                    *schema_prefix, self._table + "_v1"
                )
                old_embeddings_unqualified = sql.Identifier(self._table + "_v1")
                content = sql.Identifier(
                    *schema_prefix, self._table + self.CONTENT_TABLE
                )
                alter_table_query = sql.SQL("ALTER TABLE {t1} RENAME TO {t2}").format(
                    t1=embeddings,
                    # must supply an unqualified name for the RENAME TO target
                    t2=old_embeddings_unqualified,
                )
                cursor.execute(alter_table_query)

                self._create_table(cursor)

                insert_query = sql.SQL(
                    """
                    INSERT INTO {t1} (doc_id, embedding_id, embedding) 
                    SELECT id, embedding_id, embedding FROM {t2}
                """
                ).format(
                    t1=embeddings,
                    # reference the schema-qualified name when selecting from the old
                    # table
                    t2=old_embeddings_qualified,
                )
                cursor.execute(insert_query)

                insert_content_query = sql.SQL(
                    """
                    INSERT INTO {t1} (doc_id, text, metadata) 
                    SELECT DISTINCT id, text, metadata FROM {t2}
                """
                ).format(t1=content, t2=old_embeddings_qualified)
                cursor.execute(insert_content_query)
        except Exception as e:
            raise RuntimeError(f"Failed to migrate schema: {e}") from e

    def migrate_schema_v2_to_v3(self) -> None:
        """Migrate schema from v2 to v3.

        Difference: in the content table the `metadata` column changed from TEXT
        to JSONB.
        This routine follows the same pattern as `migrate_schema_v1_to_v2`:
        - rename the existing embeddings/content tables to a `_v2` suffix (using
          an unqualified target for the RENAME TO),
        - recreate the v3 tables via `_create_table`,
        - copy/convert data into the new tables, casting `metadata` from TEXT -> JSONB.
        """
        from psycopg2 import sql

        try:
            with self.connection.get_cursor() as cursor:
                schema_prefix = (self._schema,) if self._schema else ()

                old_content_qualified = sql.Identifier(
                    *schema_prefix, self._table + self.CONTENT_TABLE
                )

                alter_table_query = sql.SQL("ALTER TABLE {t1} RENAME TO {t2}").format(
                    t1=old_content_qualified,
                    t2=sql.Identifier(self._table + self.CONTENT_TABLE + "_v2"),
                )
                try:
                    self.logger.info(
                        "Renaming content table %s to %s",
                        old_content_qualified.as_string(cursor),
                        (self._table + self.CONTENT_TABLE + "_v2"),
                    )
                    cursor.execute(alter_table_query)
                    self.logger.debug("ALTER TABLE rename executed successfully")
                except Exception as e:
                    self.logger.exception("ALTER TABLE RENAME TO failed: %s", e)
                    raise

                try:
                    old_constraint_name = (
                        self._table + self.CONTENT_TABLE + "_pk_doc_id"
                    )
                    new_constraint_name = old_constraint_name + "_v2"
                    renamed_table_noschema = sql.Identifier(
                        self._table + self.CONTENT_TABLE + "_v2"
                    )

                    rename_constraint_sql = sql.SQL(
                        "ALTER TABLE {t} RENAME CONSTRAINT {old} TO {new}"
                    ).format(
                        t=renamed_table_noschema,
                        old=sql.Identifier(old_constraint_name),
                        new=sql.Identifier(new_constraint_name),
                    )
                    self.logger.debug(
                        "Attempting to rename constraint %s -> %s",
                        old_constraint_name,
                        new_constraint_name,
                    )
                    cursor.execute(rename_constraint_sql)
                    self.logger.debug("Constraint rename executed successfully")
                except Exception:
                    self.logger.exception(
                        "Failed to rename old primary key constraint; "
                        "continuing migration"
                    )

                # Recreate the v3 tables (this will create the content table with
                # JSONB metadata)
                self._create_table(cursor)

                import json

                from psycopg2.extras import Json, execute_values

                content = sql.Identifier(
                    *schema_prefix, self._table + self.CONTENT_TABLE
                )
                old_content_v2 = sql.Identifier(
                    *schema_prefix, self._table + self.CONTENT_TABLE + "_v2"
                )

                select_sql = sql.SQL("SELECT doc_id, text, metadata FROM {t}").format(
                    t=old_content_v2
                )
                batch_size = 1000
                named_cursor = cursor.connection.cursor(name="yb_migrate_v2_to_v3")
                try:
                    named_cursor.execute(select_sql.as_string(cursor))

                    while True:
                        rows = named_cursor.fetchmany(batch_size)
                        if not rows:
                            break

                        to_insert = []
                        for doc_id, text, metadata in rows:
                            if metadata is None or (
                                isinstance(metadata, str) and metadata.strip() == ""
                            ):
                                md = {}
                            else:
                                if not isinstance(metadata, str):
                                    md = metadata
                                else:
                                    try:
                                        md = json.loads(metadata)
                                    except Exception:
                                        md = {"old_metadata": metadata}
                            to_insert.append((doc_id, text, Json(md)))

                        if to_insert:
                            insert_sql = sql.SQL(
                                "INSERT INTO {t} (doc_id, text, metadata) VALUES %s"
                            ).format(t=content)
                            execute_values(
                                cursor,
                                insert_sql.as_string(cursor),
                                to_insert,
                                page_size=100,
                            )
                finally:
                    try:
                        named_cursor.close()
                    except Exception:
                        pass
        except Exception as e:
            raise RuntimeError(f"Failed to migrate v2 to v3: {e}") from e

    def create_index(self, index_params: Yellowbrick.IndexParams) -> None:
        """Create index from existing vectors"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            with self.connection.get_cursor() as cursor:
                self._drop_lsh_index_tables(cursor)
                self._create_lsh_index_tables(cursor)
                self._populate_hyperplanes(
                    cursor, index_params.get_param("num_hyperplanes", 128)
                )
                self._update_lsh_hashes(cursor)

        if index_params.index_type == Yellowbrick.IndexType.IVF:
            with self.connection.get_cursor() as cursor:
                self._drop_ivf_index_table(cursor)
                self._drop_ivf_centroid_tables(cursor)
                self._create_ivf_centroid_tables(cursor)
                self._create_ivf_index_table(cursor, index_params)
                self._populate_centroids(
                    cursor, index_params.get_param("num_centroids", 40)
                )
                self._k_means(
                    cursor,
                    num_centroids=index_params.get_param("num_centroids", 40),
                    max_iter=index_params.get_param("max_iter", 40),
                    threshold=index_params.get_param("threshold", 1e-4),
                )
                self._update_ivf_index(cursor, index_params)

    def drop_index(self, index_params: Yellowbrick.IndexParams) -> None:
        """Drop an index"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            with self.connection.get_cursor() as cursor:
                self._drop_lsh_index_tables(cursor)
        if index_params.index_type == Yellowbrick.IndexType.IVF:
            with self.connection.get_cursor() as cursor:
                self._drop_ivf_index_table(cursor)
                self._drop_ivf_centroid_tables(cursor)

    def _update_index(
        self, index_params: Yellowbrick.IndexParams, doc_id: uuid.UUID
    ) -> None:
        """Update an index with a new or modified embedding in the embeddings table"""
        if index_params.index_type == Yellowbrick.IndexType.LSH:
            with self.connection.get_cursor() as cursor:
                self._update_lsh_hashes(cursor, doc_id)

        if index_params.index_type == Yellowbrick.IndexType.IVF:
            with self.connection.get_cursor() as cursor:
                self._update_ivf_index(cursor, index_params, doc_id)

    def _create_lsh_index_tables(self, cursor: "PgCursor") -> None:
        """Create LSH index and hyperplane tables"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        t1 = sql.Identifier(*schema_prefix, self._table + self.LSH_INDEX_TABLE)
        t2 = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        c1 = sql.Identifier(self._table + self.LSH_INDEX_TABLE + "_pk_doc_id")
        c2 = sql.Identifier(self._table + self.LSH_INDEX_TABLE + "_fk_doc_id")
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {t1} (
                doc_id UUID NOT NULL,
                hash_index SMALLINT NOT NULL,
                hash SMALLINT NOT NULL,
                CONSTRAINT {c1} PRIMARY KEY (doc_id, hash_index),
                CONSTRAINT {c2} FOREIGN KEY (doc_id) REFERENCES {t2}(doc_id))
                DISTRIBUTE ON (doc_id) SORT ON (doc_id)
            """
            ).format(
                t1=t1,
                t2=t2,
                c1=c1,
                c2=c2,
            )
        )

        schema_prefix = (self._schema,) if self._schema else ()
        t = sql.Identifier(*schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE)
        c = sql.Identifier(self._table + self.LSH_HYPERPLANE_TABLE + "_pk_id_hp_id")
        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {t} (
                id SMALLINT NOT NULL,
                hyperplane_id SMALLINT NOT NULL,
                hyperplane FLOAT NOT NULL,
                CONSTRAINT {c} PRIMARY KEY (id, hyperplane_id))
                DISTRIBUTE REPLICATE SORT ON (id)
            """
            ).format(
                t=t,
                c=c,
            )
        )

    def _drop_lsh_index_tables(self, cursor: "PgCursor") -> None:
        """Drop LSH index tables"""
        self.drop(
            schema=self._schema, table=self._table + self.LSH_INDEX_TABLE, cursor=cursor
        )
        self.drop(
            schema=self._schema,
            table=self._table + self.LSH_HYPERPLANE_TABLE,
            cursor=cursor,
        )

    def _update_lsh_hashes(
        self,
        cursor: "PgCursor",
        doc_id: Optional[uuid.UUID] = None,
    ) -> None:
        """Add hashes to LSH index"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        lsh_hyperplane_table = sql.Identifier(
            *schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE
        )
        lsh_index_table_id = sql.Identifier(
            *schema_prefix, self._table + self.LSH_INDEX_TABLE
        )
        embedding_table_id = sql.Identifier(*schema_prefix, self._table)
        query_prefix_id = sql.SQL("INSERT INTO {}").format(lsh_index_table_id)
        condition = (
            sql.SQL("WHERE e.doc_id = {doc_id}").format(doc_id=sql.Literal(str(doc_id)))
            if doc_id
            else sql.SQL("")
        )
        group_by = sql.SQL("GROUP BY 1, 2")

        input_query = sql.SQL(
            """
            {query_prefix}
            SELECT
                e.doc_id as doc_id,
                h.id as hash_index,
                CASE WHEN SUM(e.embedding * h.hyperplane) > 0 THEN 1 ELSE 0 END as hash
            FROM {embedding_table} e
            INNER JOIN {hyperplanes} h ON e.embedding_id = h.hyperplane_id
            {condition}
            {group_by}
        """
        ).format(
            query_prefix=query_prefix_id,
            embedding_table=embedding_table_id,
            hyperplanes=lsh_hyperplane_table,
            condition=condition,
            group_by=group_by,
        )
        cursor.execute(input_query)

    def _generate_tmp_lsh_hashes(
        self, cursor: "PgCursor", tmp_embedding_table: str, tmp_hash_table: str
    ) -> None:
        """Generate temp LSH"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        lsh_hyperplane_table = sql.Identifier(
            *schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE
        )
        tmp_embedding_table_id = sql.Identifier(tmp_embedding_table)
        tmp_hash_table_id = sql.Identifier(tmp_hash_table)
        query_prefix = sql.SQL("CREATE TEMPORARY TABLE {} ON COMMIT DROP AS").format(
            tmp_hash_table_id
        )
        group_by = sql.SQL("GROUP BY 1, 2")

        input_query = sql.SQL(
            """
            {query_prefix}
            SELECT
                e.doc_id,
                h.id as hash_index,
                CASE WHEN SUM(e.embedding * h.hyperplane) > 0 THEN 1 ELSE 0 END as hash
            FROM {embedding_table} e
            INNER JOIN {hyperplanes} h ON e.embedding_id = h.hyperplane_id
            {group_by}
        """
        ).format(
            query_prefix=query_prefix,
            embedding_table=tmp_embedding_table_id,
            hyperplanes=lsh_hyperplane_table,
            group_by=group_by,
        )
        cursor.execute(input_query)

    def _populate_hyperplanes(self, cursor: "PgCursor", num_hyperplanes: int) -> None:
        """Generate random hyperplanes and store in Yellowbrick"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        hyperplanes_table = sql.Identifier(
            *schema_prefix, self._table + self.LSH_HYPERPLANE_TABLE
        )
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {t}").format(t=hyperplanes_table))
        if cursor.fetchone()[0] > 0:
            return

        t = sql.Identifier(*schema_prefix, self._table)
        cursor.execute(sql.SQL("SELECT MAX(embedding_id) FROM {t}").format(t=t))
        num_dimensions = cursor.fetchone()[0]
        num_dimensions += 1

        insert_query = sql.SQL(
            """
            WITH parameters AS (
                SELECT {num_hyperplanes} AS num_hyperplanes,
                    {dims_per_hyperplane} AS dims_per_hyperplane
            )
            INSERT INTO {hyperplanes_table} (id, hyperplane_id, hyperplane)
                SELECT id, hyperplane_id, (random() * 2 - 1) AS hyperplane
                FROM
                (SELECT range-1 id FROM sys.rowgenerator
                    WHERE range BETWEEN 1 AND
                    (SELECT num_hyperplanes FROM parameters) AND
                    worker_lid = 0 AND thread_id = 0) a,
                (SELECT range-1 hyperplane_id FROM sys.rowgenerator
                    WHERE range BETWEEN 1 AND
                    (SELECT dims_per_hyperplane FROM parameters) AND
                    worker_lid = 0 AND thread_id = 0) b
        """
        ).format(
            num_hyperplanes=sql.Literal(num_hyperplanes),
            dims_per_hyperplane=sql.Literal(num_dimensions),
            hyperplanes_table=hyperplanes_table,
        )
        cursor.execute(insert_query)

    def _create_ivf_index_table(
        self, cursor: "PgCursor", index_params: IndexParams
    ) -> None:
        """Create IVF index and centroid tables"""
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        index_table = sql.Identifier(*schema_prefix, self._table + self.IVF_INDEX_TABLE)
        content_table = sql.Identifier(*schema_prefix, self._table + self.CONTENT_TABLE)
        centroid_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )
        c1 = sql.Identifier(self._table + self.IVF_INDEX_TABLE + "_pk_doc_id")
        c2 = sql.Identifier(self._table + self.IVF_INDEX_TABLE + "_fk_doc_id")

        quantization = index_params.get_param("quantization", False)
        if quantization:
            quantization_sql = sql.SQL("embedding SMALLINT NOT NULL")
        else:
            quantization_sql = sql.SQL("embedding FLOAT NOT NULL")

        index_table_sql = sql.SQL(
            """
            DROP TABLE IF EXISTS {index_table};
            CREATE TABLE {index_table} (
                id INT NOT NULL,
                doc_id UUID NOT NULL,
                embedding_id SMALLINT NOT NULL,
                {quantization_sql},
                CONSTRAINT {c1} PRIMARY KEY (id, doc_id),
                CONSTRAINT {c2} FOREIGN KEY (doc_id) REFERENCES {content_table}(doc_id)
            )
            DISTRIBUTE ON (doc_id) SORT ON (id)
        """
        ).format(
            index_table=index_table,
            content_table=content_table,
            centroid_table=centroid_table,
            quantization_sql=quantization_sql,
            c1=c1,
            c2=c2,
        )
        cursor.execute(index_table_sql)

    def _drop_ivf_index_table(self, cursor: "PgCursor") -> None:
        """Drop IVF index tables"""
        self.drop(
            schema=self._schema, table=self._table + self.IVF_INDEX_TABLE, cursor=cursor
        )
        self.drop(
            schema=self._schema,
            table=self._table + self.IVF_CENTROID_TABLE,
            cursor=cursor,
        )

    def _create_ivf_centroid_tables(self, cursor: "PgCursor") -> None:
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        centroid_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )
        # content_table = sql.Identifier(  # Unused variable
        #     *schema_prefix, self._table + self.CONTENT_TABLE
        # )
        c1 = sql.Identifier(self._table + self.IVF_CENTROID_TABLE + "_pk_doc_id")

        centroid_table_sql = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {centroid_table} (
                id INT NOT NULL,
                centroid_id SMALLINT NOT NULL,
                centroid FLOAT NOT NULL,
                CONSTRAINT {c1} PRIMARY KEY (id, centroid_id)
            )
            DISTRIBUTE REPLICATE
        """
        ).format(
            centroid_table=centroid_table,
            c1=c1,
        )
        cursor.execute(centroid_table_sql)

        new_centroid_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE + "_new"
        )
        c1 = sql.Identifier(
            self._table + self.IVF_CENTROID_TABLE + "_new" + "_pk_doc_id"
        )

        new_centroid_table_sql = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {new_centroid_table} (
                id INT NOT NULL,
                centroid_id SMALLINT NOT NULL,
                centroid FLOAT NOT NULL,
                CONSTRAINT {c1} PRIMARY KEY (id, centroid_id)
            )
            DISTRIBUTE REPLICATE
        """
        ).format(
            new_centroid_table=new_centroid_table,
            c1=c1,
        )
        cursor.execute(new_centroid_table_sql)

    def _drop_ivf_centroid_tables(self, cursor: "PgCursor") -> None:
        """Drop IVF centroid tables"""
        self.drop(
            schema=self._schema,
            table=self._table + self.IVF_CENTROID_TABLE,
            cursor=cursor,
        )
        self.drop(
            schema=self._schema,
            table=self._table + self.IVF_CENTROID_TABLE + "_new",
            cursor=cursor,
        )

    def _update_centroids(self, cursor: "PgCursor") -> None:
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        embeddings_table = sql.Identifier(*schema_prefix, self._table)
        centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )
        new_centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE + "_new"
        )

        self._create_ivf_centroid_tables(cursor)

        update_centroid_sql = sql.SQL(
            """
            SET enable_rowpacket_compression_in_distribution=TRUE;
            WITH CentroidDistances AS (
                SELECT
                    e.doc_id AS edoc_id,
                    c.id AS cdoc_id,
                    SUM(e.embedding * c.centroid) / 
                    (SQRT(SUM(c.centroid * c.centroid)) * 
                    SQRT(SUM(e.embedding * e.embedding))) AS cosine_similarity
                FROM {embeddings_table} e
                JOIN {centroids_table} c ON e.embedding_id = c.centroid_id
                GROUP BY edoc_id, cdoc_id   
            ),  
            MaxSimilarities AS (
                SELECT
                    edoc_id,
                    cdoc_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY edoc_id ORDER BY cosine_similarity DESC
                    ) AS rank
                FROM CentroidDistances
            ),
            AssignedClusters AS (
            SELECT
                edoc_id,
                cdoc_id
                FROM MaxSimilarities
                WHERE rank = 1
            ),
            ClusterAverages AS (
                SELECT
                    ac.cdoc_id AS id,
                    e.embedding_id AS centroid_id,
                    AVG(e.embedding) AS centroid
                FROM AssignedClusters ac
                JOIN {embeddings_table} e ON ac.edoc_id = e.doc_id
                GROUP BY ac.cdoc_id, e.embedding_id
            )
            INSERT INTO {new_centroids_table}
            SELECT
                ca.id,
                ca.centroid_id,
                ca.centroid
                FROM ClusterAverages ca
                ORDER BY 1 ASC
            """
        ).format(
            centroids_table=centroids_table,
            new_centroids_table=new_centroids_table,
            embeddings_table=embeddings_table,
        )
        cursor.execute(update_centroid_sql)

    def _centroid_shift(self, cursor: "PgCursor") -> float:
        from psycopg2 import sql

        max_shift = float("inf")
        schema_prefix = (self._schema,) if self._schema else ()
        centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )
        centroids_table_noschema = sql.Identifier(self._table + self.IVF_CENTROID_TABLE)

        new_centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE + "_new"
        )
        centroid_shift_sql = sql.SQL(
            """
            WITH CentroidPairs AS (
            SELECT
                c1.id AS centroid1,
                c2.id AS centroid2,
                c1.centroid_id AS dim,
                c1.centroid,
                c2.centroid,
                (c1.centroid - c2.centroid) * (c1.centroid - c2.centroid) AS sq_diff
                FROM {centroids_table} c1
                JOIN {new_centroids_table} c2
                ON c1.centroid_id = c2.centroid_id AND c1.id = c2.id
            )
            SELECT MAX(euclidean_distance) AS max_shift from (
                SELECT
                    centroid1,
                    centroid2,
                    SQRT(SUM(sq_diff)) AS euclidean_distance
                FROM CentroidPairs
                GROUP BY 1,2
            ) shifts
        """
        ).format(
            centroids_table=centroids_table, new_centroids_table=new_centroids_table
        )
        cursor.execute(centroid_shift_sql)
        max_shift = cursor.fetchone()[0]

        c1 = sql.Identifier(
            self._table + self.IVF_CENTROID_TABLE + "_new" + "_pk_doc_id"
        )
        c2 = sql.Identifier(self._table + self.IVF_CENTROID_TABLE + "_pk_doc_id")
        swap_sql = sql.SQL(
            """
            DROP TABLE {centroids_table} CASCADE;
            ALTER TABLE {new_centroids_table} DROP CONSTRAINT {c1};
            ALTER TABLE {new_centroids_table} RENAME TO {centroids_table_noschema};
            ALTER TABLE {centroids_table} ADD CONSTRAINT {c2} PRIMARY KEY (id);
        """
        ).format(
            centroids_table=centroids_table,
            new_centroids_table=new_centroids_table,
            c1=c1,
            c2=c2,
            centroids_table_noschema=centroids_table_noschema,
        )
        cursor.execute(swap_sql)

        return max_shift

    def _populate_centroids(self, cursor: "PgCursor", num_centroids: int) -> None:
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )

        t = sql.Identifier(*schema_prefix, self._table)
        cursor.execute(sql.SQL("SELECT MAX(embedding_id) FROM {t}").format(t=t))
        num_dimensions = cursor.fetchone()[0]
        num_dimensions += 1

        centroids_insert_sql = sql.SQL(
            """
            WITH parameters AS (
                SELECT {num_centroids} AS num_centroids,
                    {dims_per_centroid} AS dims_per_centroid
            )
            INSERT INTO {centroids_table} (id, centroid_id, centroid)
                SELECT id, centroid_id, (random() * 2 - 1) AS centroid
                FROM
                (SELECT range-1 id FROM sys.rowgenerator
                    WHERE range BETWEEN 1 AND
                    (SELECT num_centroids FROM parameters) AND
                     worker_lid = 0 AND thread_id = 0) a,
                (SELECT range-1 centroid_id FROM sys.rowgenerator
                    WHERE range BETWEEN 1 AND
                (SELECT dims_per_centroid FROM parameters) AND
                    worker_lid = 0 AND thread_id = 0) b
                ORDER BY 1 ASC
         """
        ).format(
            num_centroids=sql.Literal(num_centroids),
            dims_per_centroid=sql.Literal(num_dimensions),
            centroids_table=centroids_table,
        )
        cursor.execute(centroids_insert_sql)

    def _k_means(
        self, cursor: "PgCursor", num_centroids: int, max_iter: int = 10, threshold: Optional[float] = 1e-4
    ) -> None:
        self._populate_centroids(cursor, num_centroids)

        for _ in range(max_iter):
            self._update_centroids(cursor)
            max_shift = self._centroid_shift(cursor)
            if threshold is not None and max_shift < threshold:
                break

    def _update_ivf_index(
        self,
        cursor: "PgCursor",
        index_params: IndexParams,
        doc_id: Optional[uuid.UUID] = None,
    ) -> None:
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        embeddings_table = sql.Identifier(*schema_prefix, self._table)
        ivf_index_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_INDEX_TABLE
        )
        centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )
        quantization = index_params.get_param("quantization", False)
        if quantization:
            quantization_sql = sql.SQL(
                "ROUND(((e.embedding + 1) / 2) * 255) as embedding"
            )
        else:
            quantization_sql = sql.SQL("e.embedding")

        if doc_id:
            # Ensure doc_id is safely composed into the SQL using a Literal
            where_sql = sql.SQL("WHERE edoc_id = {}")
            where_sql = where_sql.format(sql.Literal(str(doc_id)))
        else:
            where_sql = sql.SQL("WHERE 1=1")
        insert_index_sql = sql.SQL(
            """
            SET enable_rowpacket_compression_in_distribution=TRUE;
            WITH CentroidDistances AS (
                SELECT
                    e.doc_id AS edoc_id,
                    c.id AS cdoc_id,
                    SUM(e.embedding * c.centroid) / 
                    (SQRT(SUM(c.centroid * c.centroid)) * 
                    SQRT(SUM(e.embedding * e.embedding))) AS cosine_similarity
                FROM {embeddings_table} e
                JOIN {centroids_table} c ON e.embedding_id = c.centroid_id
                {where_sql}
                GROUP BY edoc_id, cdoc_id   
            ),  
            MaxSimilarities AS (
                SELECT
                    edoc_id,
                    cdoc_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY edoc_id ORDER BY cosine_similarity DESC
                    ) AS rank
                FROM CentroidDistances
            )
            INSERT INTO {ivf_index_table}
            SELECT
                cdoc_id,
                edoc_id,
                e.embedding_id,
                {quantization_sql}
            FROM MaxSimilarities ms
            JOIN {embeddings_table} e ON e.doc_id = ms.edoc_id 
            WHERE ms.rank = 1
            ORDER BY cdoc_id, edoc_id ASC
        """
        ).format(
            ivf_index_table=ivf_index_table,
            embeddings_table=embeddings_table,
            centroids_table=centroids_table,
            where_sql=where_sql,
            quantization_sql=quantization_sql,
        )
        cursor.execute(insert_index_sql)

    def _find_centroid(self, cursor: "PgCursor", query_embedding_table: str) -> int:
        from psycopg2 import sql

        schema_prefix = (self._schema,) if self._schema else ()
        embedding_table = sql.Identifier(query_embedding_table)
        ivf_index_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_INDEX_TABLE
        )
        centroids_table = sql.Identifier(
            *schema_prefix, self._table + self.IVF_CENTROID_TABLE
        )
        search_index_sql = sql.SQL(
            """
            SET enable_rowpacket_compression_in_distribution=TRUE;

            WITH CentroidDistances AS (
                SELECT
                    e.doc_id AS edoc_id,
                    c.id AS cdoc_id,
                    SUM(e.embedding * c.centroid) / 
                    (SQRT(SUM(c.centroid * c.centroid)) * 
                    SQRT(SUM(e.embedding * e.embedding))) AS cosine_similarity
                FROM {embedding_table} e
                JOIN {centroids_table} c ON e.embedding_id = c.centroid_id
                GROUP BY edoc_id, cdoc_id   
            ),  
            MaxSimilarities AS (
                SELECT
                    edoc_id,
                    cdoc_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY edoc_id ORDER BY cosine_similarity DESC
                    ) AS rank
                FROM CentroidDistances
            )
            SELECT
                cdoc_id
            FROM MaxSimilarities
            WHERE rank = 1
        """
        ).format(
            ivf_index_table=ivf_index_table,
            embedding_table=embedding_table,
            centroids_table=centroids_table,
        )
        cursor.execute(search_index_sql)
        return cursor.fetchone()[0]


def jsonFilter2sqlWhere(
    filter_dict: Dict[str, Any], metadata_column: str = "metadata"
) -> str:
    """
    Convert Pinecone filter syntax to Yellowbrick SQL WHERE clause using
    JSON path syntax.

    Args:
        filter_dict: Pinecone-style filter dictionary
        metadata_column: Name of the JSONB column containing metadata
            (default: "metadata")

    Returns:
        SQL WHERE clause string using Yellowbrick JSON path syntax

    Example:
        filter = {"genre": {"$eq": "documentary"}, "year": {"$gte": 2020}}
        result = jsonFilter2sqlWhere(filter)
        # Returns: "(metadata:$.genre::TEXT = 'documentary' AND
        #    metadata:$.year::INTEGER >= 2020)"
    """
    if not filter_dict:
        return "1=1"  # No filter condition

    return _process_filter_dict(filter_dict, metadata_column)


def _process_filter_dict(filter_dict: Dict[str, Any], metadata_column: str) -> str:
    """Process a filter dictionary and return SQL WHERE clause."""
    conditions = []

    for key, value in filter_dict.items():
        if key == "$and":
            and_conditions = []
            for condition in value:
                and_conditions.append(_process_filter_dict(condition, metadata_column))
            conditions.append(f"({' AND '.join(and_conditions)})")

        elif key == "$or":
            or_conditions = []
            for condition in value:
                or_conditions.append(_process_filter_dict(condition, metadata_column))
            conditions.append(f"({' OR '.join(or_conditions)})")

        else:
            # Regular field condition
            field_condition = _process_field_condition(key, value, metadata_column)
            conditions.append(field_condition)

    if len(conditions) == 1:
        return conditions[0]
    else:
        # Multiple conditions at same level are implicitly AND
        return f"({' AND '.join(conditions)})"


def _process_field_condition(
    field_name: str, condition: Any, metadata_column: str
) -> str:
    """Process a single field condition."""

    # Handle simple equality (shorthand syntax)
    if not isinstance(condition, dict):
        return _create_json_condition(field_name, "$eq", condition, metadata_column)

    # Handle operator-based conditions
    conditions = []
    for operator, value in condition.items():
        sql_condition = _create_json_condition(
            field_name, operator, value, metadata_column
        )
        conditions.append(sql_condition)

    if len(conditions) == 1:
        return conditions[0]
    else:
        # Multiple operators on same field are implicitly AND
        return f"({' AND '.join(conditions)})"


def _create_json_condition(
    field_name: str, operator: str, value: Any, metadata_column: str
) -> str:
    """Create a single JSON condition using Yellowbrick JSON path syntax."""

    # Escape field name for JSON path if it contains special characters
    escaped_field = _escape_json_field_name(field_name)
    json_path = f"{metadata_column}:$.{escaped_field}"

    # Determine the cast type for the field
    cast_type = (
        f"::{_get_cast_type(value)}" if not isinstance(value, bool) else "::BOOLEAN"
    )

    if operator == "$eq":
        return f"{json_path}{cast_type} = {_format_sql_value(value)}"

    elif operator == "$ne":
        return f"{json_path}{cast_type} != {_format_sql_value(value)}"

    elif operator == "$gt":
        return f"{json_path}{cast_type} > {_format_sql_value(value)}"

    elif operator == "$gte":
        return f"{json_path}{cast_type} >= {_format_sql_value(value)}"

    elif operator == "$lt":
        return f"{json_path}{cast_type} < {_format_sql_value(value)}"

    elif operator == "$lte":
        return f"{json_path}{cast_type} <= {_format_sql_value(value)}"

    elif operator == "$in":
        if not isinstance(value, list):
            raise ValueError(f"$in operator requires a list, got {type(value)}")

        # Use Yellowbrick's supported IN syntax
        formatted_values = ", ".join(_format_sql_value(v) for v in value)
        return f"{json_path}{cast_type} IN ({formatted_values})"

    elif operator == "$nin":
        if not isinstance(value, list):
            raise ValueError(f"$nin operator requires a list, got {type(value)}")

        # For NOT IN operations, convert to AND of != conditions
        nin_conditions = [
            f"{json_path}{cast_type} != {_format_sql_value(v)}" for v in value
        ]
        return f"({' AND '.join(nin_conditions)})"

    elif operator == "$exists":
        base_json_path = f"{metadata_column}:$.{escaped_field}"
        if value:
            return f"{base_json_path} NULL ON ERROR IS NOT NULL"
        else:
            return f"{base_json_path} NULL ON ERROR IS NULL"

    else:
        raise ValueError(f"Unsupported operator: {operator}")


def _escape_json_field_name(field_name: str) -> str:
    """
    Escape field names for JSON path expressions in Yellowbrick.
    Uses bracket notation for fields with special characters.
    """
    # Check if field name contains special characters that need bracket notation
    special_chars = [
        ".",
        " ",
        "'",
        '"',
        "[",
        "]",
        ":",
        "-",
        "+",
        "*",
        "/",
        "\\",
        "(",
        ")",
        "{",
        "}",
    ]

    if any(char in field_name for char in special_chars):
        # Use bracket notation and escape quotes
        escaped = field_name.replace("'", "''")
        return f"['{escaped}']"
    else:
        # Use dot notation for simple field names
        return field_name


def _get_cast_type(value: Any) -> str:
    """Determine the appropriate SQL cast type based on Python value type."""
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "DOUBLE PRECISION"
    elif isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, str):
        return "TEXT"
    else:
        return "TEXT"  # Default to TEXT for other types


def _format_sql_value(value: Any) -> str:
    """Format a Python value for SQL."""
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    else:
        # For other types, convert to JSON string
        return f"'{json.dumps(value)}'"
