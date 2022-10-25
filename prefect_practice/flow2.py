from prefect import flow
import requests

@flow
def call_api(url):
    return requests.get(url).json()

api_result = call_api("http://time.jsontest.com/")

print(api_result)
