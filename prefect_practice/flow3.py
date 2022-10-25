from prefect import flow, task
import requests

@task
def call_api(url):
    response = requests.get(url)
    print(response.status_code)
    return response.json()

@flow
def api_flow(url):
    fact_json = call_api(url)
    return fact_json

api_result = api_flow("https://catfact.ninja/fact")

print(api_result)
