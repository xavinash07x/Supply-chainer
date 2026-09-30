import os
import unittest
from unittest.mock import patch
os.environ.setdefault('DEMO_MODE', 'true')
from fastapi.testclient import TestClient
from backend.main import app, predictor


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_status_reports_actual_model_state(self):
        with patch.object(predictor, 'is_trained', False):
            self.assertFalse(self.client.get('/api/status').json()['ml_trained'])

    def test_scenarios_and_validation(self):
        ids = [s['id'] for s in self.client.get('/api/scenarios').json()]
        self.assertIn('SUEZ_BLOCK', ids)
        response = self.client.post('/api/recommend', json={'source': 'Shanghai', 'destination': 'Rotterdam', 'scenario': 'TYPO'})
        self.assertIn('error', response.json())
        self.assertEqual(self.client.post('/api/recommend', json={}).status_code, 422)

    def test_city_route_request_reconciles_and_reroutes(self):
        payload = {'source': 'Shanghai', 'destination': 'Rotterdam', 'transport_preference': 'sea'}
        results = []
        for scenario in [None, 'SUEZ_BLOCK', None]:
            response = self.client.post('/api/recommend', json={**payload, 'scenario': scenario})
            self.assertEqual(response.status_code, 200)
            result = response.json()['recommendations'][0]
            self.assertEqual(round(sum(result['audit_trace']['eta'].values()), 1), result['adjusted_eta'])
            self.assertEqual(round(sum(result['audit_trace']['cost'].values()), 2), result['total_cost'])
            self.assertTrue(any(l['delay_prediction'] for l in result['legs']))
            results.append(result)
        self.assertIn('CHOKE-SUEZ', [l['to'] for l in results[0]['legs']])
        self.assertNotIn('CHOKE-SUEZ', [l['to'] for l in results[1]['legs']])
        self.assertEqual(results[0], results[2])

    def test_dashboard_hub_selection_matches_city_request(self):
        response = self.client.post('/api/recommend', json={'source': 'HUB-SHANGHAI',
            'destination': 'HUB-ROTTERDAM', 'transport_preference': 'sea', 'scenario': 'SUEZ_BLOCK'})
        self.assertEqual(response.status_code, 200)
        result = response.json()['recommendations'][0]
        self.assertIn('Port of Durban', [l['to_name'] for l in result['legs']])
        self.assertNotIn('CHOKE-SUEZ', [l['to'] for l in result['legs']])


if __name__ == '__main__': unittest.main()
