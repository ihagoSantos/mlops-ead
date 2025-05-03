from locust import HttpUser, task, between

class ApiLoadRunner(HttpUser):
    wait_time = between(0.5, 2.5)

    @task
    def request(self):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        request_body = {
            "accelerations": 0,
            "fetal_movement": 0,
            'uterine_contractions': 0,
            'severe_decelerations': 0,
        }

        self.client.post(
            url='/predict',
            headers=headers,
            json=request_body
        )