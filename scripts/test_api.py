import argparse
import requests


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test movie genre prediction API")
    parser.add_argument(
        "--uri",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="API endpoint URI (default: http://127.0.0.1:8000/predict)",
    )
    args = parser.parse_args()

    try:
        # Send POST request to API
        response = requests.post(
            args.uri,
            json={
                "overview": "Imprisoned in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope."
            },
        )

        # Check response
        if response.status_code == 200:
            print("Success!")
            print("Response:", response.json())
            print("Expected:", {"Drama": 1.0, "Crime": 1.0})
        else:
            print(f"Error: Status code {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")


if __name__ == "__main__":
    main()
