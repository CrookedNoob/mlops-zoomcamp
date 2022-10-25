from prefect import flow, task
import requests

@task
def call_api(url):
    response = requests.get(url)
    print(response.status_code)
    return response.json()


@task
def parse_fact(response):
    fact = response["fact"]
    print(fact)
    return fact

@flow
def api_flow(url):
    fact_json = call_api(url)
    fact_text = parse_fact(fact_json)
    return fact_json

api_result = api_flow("https://catfact.ninja/fact")

print(api_result)
