import json
import logging
from typing import List, Optional

import pytest

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import Yellowbrick
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
    fake_texts,
)

YELLOWBRICK_URL = "postgresql://[USERNAME]:[PASSWORD]@[HOSTNAME]:5432/[DATABASE]"
YELLOWBRICK_SCHEMA = "[SCHEMA]"

YELLOWBRICK_TABLE = "my_embeddings"
YELLOWBRICK_CONTENT = "my_embeddings_content"


def _yellowbrick_vector_from_texts(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> Yellowbrick:
    db = Yellowbrick.from_texts(
        fake_texts,
        ConsistentFakeEmbeddings(),
        metadatas,
        YELLOWBRICK_URL,
        table=YELLOWBRICK_TABLE,
        schema=YELLOWBRICK_SCHEMA,
        drop=drop,
    )
    db.logger.setLevel(logging.DEBUG)
    return db


def _yellowbrick_vector_from_texts_no_schema(
    metadatas: Optional[List[dict]] = None, drop: bool = True
) -> Yellowbrick:
    db = Yellowbrick.from_texts(
        fake_texts,
        ConsistentFakeEmbeddings(),
        metadatas,
        YELLOWBRICK_URL,
        table=YELLOWBRICK_TABLE,
        drop=drop,
    )
    db.logger.setLevel(logging.DEBUG)
    return db


@pytest.mark.requires("yb-vss")
def test_yellowbrick() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_text() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts)
        output = docsearch.similarity_search("oof", k=1)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_delete() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        added_docs = docsearch.add_texts(texts)
        output = docsearch.similarity_search("oof", k=1)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(added_docs)
        output = docsearch.similarity_search("oof", k=1)
        assert output != [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_delete_all() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        output = docsearch.similarity_search("foo", k=1)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts)
        output = docsearch.similarity_search("oof", k=1)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(delete_all=True)
        output = docsearch.similarity_search("oof", k=1)
        assert output != [Document(page_content="oof", metadata={})]
        output = docsearch.similarity_search("foo", k=1)
        assert output != [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_search() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_search_update() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_delete() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        added_docs = docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(added_docs)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output != [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_lsh_delete_all() -> None:
    """Test end to end construction and search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.LSH, {"num_hyperplanes": 10, "hamming_distance": 0}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(delete_all=True)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output != [Document(page_content="oof", metadata={})]
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output != [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_with_score() -> None:
    """Test end to end construction and search with scores and IDs."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _yellowbrick_vector_from_texts(metadatas=metadatas)
        output = docsearch.similarity_search_with_score("foo", k=3)
        docs = [o[0] for o in output]
        distances = [o[1] for o in output]
        assert docs == [
            Document(page_content="foo", metadata={"page": 0}),
            Document(page_content="bar", metadata={"page": 1}),
            Document(page_content="baz", metadata={"page": 2}),
        ]
        assert distances[0] > distances[1] > distances[2]


@pytest.mark.requires("yb-vss")
def test_yellowbrick_ivf_search() -> None:
    """Test end to end construction and search for IVF index."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.IVF, {"num_centroids": 5, "quantization": False}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        # Ensure index tables are cleaned up
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_ivf_search_update() -> None:
    """Test end to end construction and search with updates for IVF index."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.IVF, {"num_centroids": 5, "quantization": False}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_ivf_delete() -> None:
    """Test end to end construction and delete for IVF index."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.IVF, {"num_centroids": 5, "quantization": False}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        added_docs = docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(added_docs)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output != [Document(page_content="oof", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_ivf_delete_all() -> None:
    """Test end to end construction and delete_all for IVF index."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        index_params = Yellowbrick.IndexParams(
            Yellowbrick.IndexType.IVF, {"num_centroids": 5, "quantization": False}
        )
        docsearch.drop_index(index_params)
        docsearch.create_index(index_params)
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output == [Document(page_content="foo", metadata={})]
        texts = ["oof"]
        docsearch.add_texts(texts, index_params=index_params)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output == [Document(page_content="oof", metadata={})]
        docsearch.delete(delete_all=True)
        output = docsearch.similarity_search("oof", k=1, index_params=index_params)
        assert output != [Document(page_content="oof", metadata={})]
        output = docsearch.similarity_search("foo", k=1, index_params=index_params)
        assert output != [Document(page_content="foo", metadata={})]
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)
        docsearch.drop_index(index_params=index_params)


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_extra() -> None:
    """Test end to end construction and MRR search."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        texts = ["foo", "bar", "baz"]
        metadatas = [{"page": i} for i in range(len(texts))]
        docsearch = _yellowbrick_vector_from_texts(metadatas=metadatas)
        docsearch.add_texts(texts, metadatas)
        output = docsearch.similarity_search("foo", k=10)
        assert len(output) == 6


@pytest.mark.requires("yb-vss")
def test_yellowbrick_add_text_filter() -> None:
    """Test adding texts with metadata and filtering via similarity_search filter
    argument."""
    docsearches = [
        _yellowbrick_vector_from_texts(),
        _yellowbrick_vector_from_texts_no_schema(),
    ]
    for docsearch in docsearches:
        # Add texts with various metadata for testing different filters
        texts = [
            "unique-filter-text-1",
            "unique-filter-text-2",
            "unique-filter-text-3",
            "unique-filter-text-4",
            "unique-filter-text-5",
        ]
        metadatas = [
            {"category": "special", "priority": 1, "tags": ["important", "urgent"]},
            {"category": "normal", "priority": 2, "tags": ["important"]},
            {"category": "special", "priority": 3, "active": True},
            {"category": "normal", "priority": 4, "active": False},
            {"category": "archived", "tags": ["old"]},
        ]
        added_ids = docsearch.add_texts(texts, metadatas)

        # Basic search without filter
        output = docsearch.similarity_search("unique-filter-text", k=5)
        assert len(output) == 5
        
        # Test $eq operator
        eq_filter = {"category": {"$eq": "special"}}
        output_eq = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(eq_filter)
        )
        assert len(output_eq) == 2
        assert all(d.page_content.startswith("unique-filter-text") for d in output_eq)
        assert all(
            i in [1, 3] for i in [int(d.page_content[-1]) for d in output_eq]
        )
        
        # Test $ne operator
        ne_filter = {"category": {"$ne": "special"}}
        output_ne = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(ne_filter)
        )
        assert len(output_ne) == 3
        assert all(
            i in [2, 4, 5] for i in [int(d.page_content[-1]) for d in output_ne]
        )
        
        # Test $gt operator
        gt_filter = {"priority": {"$gt": 2}}
        output_gt = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(gt_filter)
        )
        assert len(output_gt) == 2
        assert all(
            i in [3, 4] for i in [int(d.page_content[-1]) for d in output_gt]
        )
        
        # Test $in operator
        in_filter = {"category": {"$in": ["special", "archived"]}}
        output_in = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(in_filter)
        )
        assert len(output_in) == 3
        assert all(
            i in [1, 3, 5] for i in [int(d.page_content[-1]) for d in output_in]
        )
        
        # Test $nin operator
        nin_filter = {"category": {"$nin": ["normal"]}}
        output_nin = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(nin_filter)
        )
        assert len(output_nin) == 3
        assert all(
            i in [1, 3, 5] for i in [int(d.page_content[-1]) for d in output_nin]
        )
        
        # Test $exists operator
        exists_filter = {"active": {"$exists": True}}
        output_exists = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(exists_filter)
        )
        assert len(output_exists) == 2
        assert all(
            i in [3, 4] for i in [int(d.page_content[-1]) for d in output_exists]
        )
        
        # Test $and operator
        and_filter = {"$and": [{"category": "special"}, {"priority": {"$lt": 3}}]}
        output_and = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(and_filter)
        )
        assert len(output_and) == 1
        assert output_and[0].page_content == "unique-filter-text-1"
        
        # Test $or operator
        or_filter = {"$or": [{"category": "archived"}, {"priority": 1}]}
        output_or = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(or_filter)
        )
        assert len(output_or) == 2
        assert all(
            i in [1, 5] for i in [int(d.page_content[-1]) for d in output_or]
        )
        
        # Test nested complex filter
        complex_filter = {
            "$and": [
                {"$or": [{"category": "special"}, {"category": "normal"}]},
                {"$or": [{"priority": {"$lt": 3}}, {"active": True}]}
            ]
        }
        output_complex = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(complex_filter)
        )
        assert len(output_complex) == 3
        assert all(
            i in [1, 2, 3] for i in [int(d.page_content[-1]) for d in output_complex]
        )
        
        # Test empty filter (should be equivalent to no filter)
        empty_filter = {}
        output_empty = docsearch.similarity_search(
            "unique-filter-text", k=5, filter=json.dumps(empty_filter)
        )
        assert len(output_empty) == 5
        
        # Clean up
        docsearch.delete(added_ids)
        docsearch.drop(table=YELLOWBRICK_TABLE, schema=docsearch._schema)
        docsearch.drop(table=YELLOWBRICK_CONTENT, schema=docsearch._schema)