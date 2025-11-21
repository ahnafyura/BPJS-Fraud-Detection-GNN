from py2neo import Graph
import pandas as pd
import os
from config import env

def export_nodes_and_edges(graph: Graph, out_dir: str):
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    nodes_query = """
    MATCH (n)
    RETURN id(n) AS node_id,
        labels(n) AS labels,
        properties(n) AS properties
    """
    nodes_df = pd.DataFrame(graph.run(nodes_query).data())
    props_df = nodes_df["properties"].apply(pd.Series)
    nodes_df = pd.concat([nodes_df.drop(columns=["properties"]), props_df], axis=1)


    nodes_df.to_csv(os.path.join(out_dir, "nodes.csv"), index=False)
    print(f"nodes.csv exported to {out_dir}")

    # --- EDGES ---
    edges_query = """
    MATCH (a)-[r]->(b)
    RETURN id(a) AS source,
           id(b) AS target,
           type(r) AS type,
           properties(r) AS properties
    """

    edges_df = pd.DataFrame(graph.run(edges_query).data())

    if not edges_df.empty and "properties" in edges_df.columns:
        props_df = edges_df["properties"].apply(pd.Series)
        edges_df = pd.concat([edges_df.drop(columns=["properties"]), props_df], axis=1)

    edges_df.to_csv(os.path.join(out_dir, "edges.csv"), index=False)
    print(f"edges.csv exported to {out_dir}")

def export_data():
    graph = Graph(env.url, auth=(env.uname, env.pw))
    export_nodes_and_edges(graph, env.DATA_PROCESSED_OUT_DIR)

if __name__ == "__main__":
    export_data()
