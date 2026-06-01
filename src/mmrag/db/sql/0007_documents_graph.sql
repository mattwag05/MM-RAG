-- Milestone 7/2.x foundation: document search projection + graph retrieval.

CREATE VIRTUAL TABLE IF NOT EXISTS fts_content_items USING fts5(
    item_id UNINDEXED,
    asset_id UNINDEXED,
    item_type UNINDEXED,
    text,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS nodes (
    id             TEXT PRIMARY KEY,
    node_type      TEXT NOT NULL,
    asset_id       TEXT REFERENCES assets(id) ON DELETE CASCADE,
    label          TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS edges (
    id              TEXT PRIMARY KEY,
    source_node_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target_node_id  TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    edge_type       TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(source_node_id, target_node_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_nodes_asset_id ON nodes(asset_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
