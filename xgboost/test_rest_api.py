from rest_api import app


def test_get_prediction():
    response = app.test_client().get(
        '/prediction?trip_seconds=885&trip_miles=3.45&pickup_community_area=null&dropoff_community_area=null&fare=12.75&tolls=0&extras=4'
    )

    assert response.status_code == 200
    assert type(response.get_data(as_text=True)) is str
