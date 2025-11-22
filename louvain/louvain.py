from py2neo import Graph
from config import env

def run_patient_claim_analysis(graph: Graph):
    # 1. Drop existing graph projection (safe for GDS 1.x)
    try:
        graph.run("CALL gds.graph.drop('patient_claim_graph')")
    except Exception as e:
        if "GraphNotFound" not in str(e):
            raise

    # 2. Project the graph using exact ETL labels
    graph.run("""
    CALL gds.graph.project(
        'patient_claim_graph',
        ['Patient','Claim','Diagnosis','Procedure','ServiceType','CareClass'],
        ['MADE_CLAIM','HAS_DIAGNOSIS','HAS_PROCEDURE','HAS_SERVICE_TYPE','IN_CLASS']
    )
    """)

    # 3. Run Louvain and store community_id (snake_case)
    graph.run("""
    CALL gds.louvain.write('patient_claim_graph', {
        includeIntermediateCommunities: true,
        writeProperty: 'community_id'
    })
    """)

    # 4. Centrality metrics (snake_case)
    graph.run("CALL gds.degree.write('patient_claim_graph', { writeProperty: 'degree' })")
    graph.run("CALL gds.pageRank.write('patient_claim_graph', { writeProperty: 'pagerank' })")
    graph.run("CALL gds.betweenness.write('patient_claim_graph', { writeProperty: 'betweenness' })")
    graph.run("CALL gds.closeness.write('patient_claim_graph', { writeProperty: 'closeness' })")

    # 5. Community size (Neo4j 4.x+ syntax)
    graph.run("""
    MATCH (n)
    WHERE n.community_id IS NOT NULL
    WITH n.community_id AS cid, count(n) AS community_size
    MATCH (m)
    WHERE m.community_id = cid
    SET m.community_size = community_size
    """)

    # 6. Community density
    graph.run("""
    MATCH (n)-[r]-()
    WHERE n.community_id IS NOT NULL
    WITH n.community_id AS cid, count(r) AS edges
    MATCH (m)
    WHERE m.community_id = cid
    WITH cid, edges, count(m) AS sz
    WITH cid, edges, sz,
         CASE WHEN sz > 1 THEN (2.0 * edges) / (sz * (sz - 1)) ELSE 0 END AS community_density
    MATCH (x)
    WHERE x.community_id = cid
    SET x.community_density = community_density
    """)

    print("Analysis completed.")


if __name__ == "__main__":
    graph = Graph(env.url, auth=(env.uname, env.pw))
    run_patient_claim_analysis(graph)
