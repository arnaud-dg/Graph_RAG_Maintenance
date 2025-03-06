from neo4j import GraphDatabase

uri = "neo4j+s://b37f20be.databases.neo4j.io"  # Mets ici l'URI correct si différent
username = "neo4j"
password = "8xjaBTuu80Rnfb2tQUXMMBC30_y0MVfxtr2tIeJERdk"

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run("RETURN 1")
        print("Connection successful:", result.single())
except Exception as e:
    print("Connection error:", e)
finally:
    driver.close()