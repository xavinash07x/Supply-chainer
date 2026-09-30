"""Offline regression tests: no RSS requests, downloads, or transformer weights."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import networkx as nx
from backend.engine.threat_intelligence import CARFFilter, ContrastiveNLPEngine, ThreatIntelligencePredictor
from backend.engine.route_recommender import RouteRecommender
from backend.engine.scenario_manager import ScenarioManager


class Tensor:
    def __init__(self, values): self.values = np.array(values)
    def cpu(self): return self
    def numpy(self): return self.values


class ThreatTests(unittest.TestCase):
    def test_relevant_news_survives_for_all_modes(self):
        for mode, news in [('sea', 'Vessels blocked the canal.'), ('air', 'Flights cancelled at airports.'),
                           ('rail', 'Rail tracks closed.'), ('road', 'Highway bridge collapse.')]:
            with self.subTest(mode=mode):
                self.assertEqual(CARFFilter().apply_filter(.8, news, mode.upper()), .8)

    def test_irrelevant_modes_are_filtered(self):
        for mode in ['sea', 'rail', 'road']:
            self.assertEqual(CARFFilter().apply_filter(.8, 'Airport flight cancellation.', mode), 0)
        self.assertEqual(CARFFilter().apply_filter(.8, 'Vessel grounding at port.', 'air'), 0)
        self.assertEqual(CARFFilter().apply_filter(.8, 'Normal operations.', 'sea'), 0)

    def test_mixed_mode_news_keeps_both_relevant_modes(self):
        news = 'Storm closes airport and port.'
        for mode in ['air', 'sea']:
            self.assertEqual(CARFFilter().apply_filter(.8, news, mode), .8)

    def engine(self, disasters, safe):
        engine = ContrastiveNLPEngine(lazy_load=True)
        engine._ready = True
        engine.model = unittest.mock.Mock()
        engine.util = unittest.mock.Mock()
        engine.util.cos_sim.side_effect = [Tensor(disasters), Tensor(safe)]
        engine.disaster_matrix = 'disasters'; engine.safe_matrix = 'safe'
        return engine

    def test_strong_threat_is_positive(self):
        self.assertGreater(self.engine([[.9]], [[.1]]).get_semantic_score('Port closed.'), 0)

    def test_safe_news_and_noise_are_zero(self):
        for disaster, safe in [(.1, .9), (.22, .2)]:
            self.assertEqual(self.engine([[disaster]], [[safe]]).get_semantic_score('Port open.'), 0)

    def test_safe_chunk_does_not_erase_disaster_chunk(self):
        engine = self.engine([[.9], [.3]], [[.1], [.95]])
        self.assertGreater(engine.get_semantic_score('x' * 300), 0)


class FakePredictor:
    def __init__(self): self.calls = []; self.delay = 0
    def predict_worst_case_delays(self, requests):
        self.calls.append(copy.deepcopy(requests))
        return [{'final_delay_presented': self.delay if r['destination'] == 'Suez Canal' else 0,
                 'prediction_source': 'ML_P85', 'calibration_reason': 'Test model', 'p_quantile': .85}
                for r in requests]


def engine_with_graph():
    predictor = FakePredictor()
    route = RouteRecommender.__new__(RouteRecommender)
    route.predictor = predictor; route.scenario_mgr = ScenarioManager()
    route.resolver = unittest.mock.Mock()
    route.resolver.resolve_node_to_entry_point.side_effect = lambda name: {'id': name}
    graph = nx.DiGraph()
    for node, physical, name, mode in [('S', 'SOURCE', 'Source', 'sea'),
             ('C', 'CHOKE-SUEZ', 'Suez Canal', 'sea'), ('D', 'DEST', 'Destination', 'sea'),
             ('X', 'CAPE', 'Cape', 'sea')]:
        graph.add_node(node, physical_id=physical, display_name=name, mode=mode)
    for u, v, hours in [('S', 'C', 10), ('C', 'D', 10), ('S', 'X', 30), ('X', 'D', 30)]:
        graph.add_edge(u, v, baseline_time=hours, cost=100, transport_mode='sea', type='transit', base_threat=0)
    route.unified_graph = graph
    return route, predictor


class RoutingTests(unittest.TestCase):
    def test_model_changes_route_not_only_display(self):
        route, predictor = engine_with_graph()
        normal = route.recommend('S', 'D')['recommendations'][0]
        self.assertIn('CHOKE-SUEZ', [l['to'] for l in normal['legs']])
        predictor.delay = 100
        result = route.recommend('S', 'D')['recommendations'][0]
        self.assertNotIn('CHOKE-SUEZ', [l['to'] for l in result['legs']])
        self.assertEqual(len(predictor.calls), 2)  # One graph batch per request, not per persona.

    def test_blockage_reroutes_and_normal_request_resets(self):
        route, predictor = engine_with_graph()
        result = route.recommend('S', 'D', scenario='SUEZ_BLOCK')['recommendations'][0]
        self.assertNotIn('CHOKE-SUEZ', [l['to'] for l in result['legs']])
        self.assertEqual(predictor.calls[0][0]['nlp_score'], 1.0)
        normal = route.recommend('S', 'D')['recommendations'][0]
        self.assertIn('CHOKE-SUEZ', [l['to'] for l in normal['legs']])
        self.assertIsNone(route.scenario_mgr.active_scenario_id)

    def test_scenario_delay_is_a_floor_not_double_counted(self):
        route, predictor = engine_with_graph()
        route.unified_graph.remove_node('X'); predictor.delay = 100
        result = route.recommend('S', 'D', scenario='SUEZ_BLOCK')['recommendations'][0]
        trace = result['audit_trace']
        self.assertEqual(result['adjusted_eta'], 260)
        self.assertEqual(trace['eta'], {'transit': 20, 'transfer': 0, 'model': 100, 'scenario': 140})
        self.assertEqual(sum(trace['cost'].values()), result['total_cost'])
        self.assertEqual(result['threat_level'], 1)
        predictor.delay = 300
        result = route.recommend('S', 'D', scenario='SUEZ_BLOCK')['recommendations'][0]
        self.assertEqual(result['adjusted_eta'], 320)
        self.assertEqual(result['audit_trace']['eta']['scenario'], 0)

    def test_sea_scenario_does_not_delay_road_or_transfer(self):
        route, predictor = engine_with_graph()
        route.unified_graph.remove_node('X')
        route.unified_graph['S']['C'].update(transport_mode='road')
        result = route.recommend('S', 'D', scenario='SUEZ_BLOCK')['recommendations'][0]
        self.assertEqual(result['adjusted_eta'], 20)
        self.assertEqual(result['threat_level'], 0)
        route.unified_graph['S']['C'].update(transport_mode='transfer', type='transfer')
        predictor.delay = 100
        result = route.recommend('S', 'D', scenario='SUEZ_BLOCK')['recommendations'][0]
        self.assertEqual(result['adjusted_eta'], 20)
        self.assertIsNone(result['legs'][0]['delay_prediction'])

    def test_audit_cost_and_time_reconcile(self):
        route, predictor = engine_with_graph(); predictor.delay = 8
        for result in route.recommend('S', 'D')['recommendations']:
            self.assertEqual(round(sum(result['audit_trace']['eta'].values()), 1), result['adjusted_eta'])
            self.assertEqual(round(sum(result['audit_trace']['cost'].values()), 2), result['total_cost'])
            self.assertEqual(sum(l['eta'] for l in result['legs']), result['adjusted_eta'])
            self.assertNotIn('vs', result['explanation'])

    def test_invalid_inputs_and_avoided_endpoint(self):
        route, _ = engine_with_graph()
        for params in [{'scenario': 'TYPO'}, {'transport_preference': 'space'}, {'routing_policy': 'TYPO'},
                       {'overrides': {'avoid_chokepoints': ['SOURCE']}}]:
            self.assertIn('error', route.recommend('S', 'D', **params))

    def test_constraints_apply_to_adjusted_time(self):
        route, predictor = engine_with_graph(); predictor.delay = 100
        self.assertIn('error', route.recommend('S', 'D', overrides={'max_delay': 1}))
        self.assertIn('error', route.recommend('S', 'D', overrides={'cost_ceiling': 50}))


class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.predictor = ThreatIntelligencePredictor()

    def test_supplied_model_predicts_and_calibrates(self):
        self.assertTrue(self.predictor.is_trained)
        result = self.predictor.predict_worst_case_delay('Shanghai', 'Rotterdam', 'sea', nlp_score=1)
        self.assertEqual(result['prediction_source'], 'ML_P85')
        self.assertEqual(result['p_quantile'], .85)
        self.assertEqual(result['unsupported_features'], [])
        self.assertGreaterEqual(result['final_delay_presented'], 20.6)
        self.assertLessEqual(result['final_delay_presented'], 360)

    def test_batch_matches_single_prediction(self):
        requests = [dict(origin='Shanghai', destination='Rotterdam', transport_mode='sea', nlp_score=score)
                    for score in [0, 1]]
        batch = self.predictor.predict_worst_case_delays(requests)
        self.assertEqual(batch, [self.predictor.predict_worst_case_delay(**r) for r in requests])
        self.assertGreater(batch[1]['final_delay_presented'], batch[0]['final_delay_presented'])

    def test_unknown_hub_is_disclosed(self):
        result = self.predictor.predict_worst_case_delay('Unknown city', 'Unknown destination', 'sea')
        self.assertEqual(set(result['unsupported_features']), {'Origin_Node', 'Destination_Node'})
        self.assertFalse(result['is_defensible'])

    def test_missing_model_and_inference_error_use_visible_prior(self):
        predictor = ThreatIntelligencePredictor(lazy_load=True)
        result = predictor.predict_worst_case_delay('Shanghai', 'Rotterdam', 'sea')
        self.assertEqual(result['prediction_source'], 'OPERATIONAL_PRIOR')
        self.assertIsNone(result['p_quantile'])
        self.assertEqual(result['final_delay_presented'], 48)
        with patch.object(self.predictor.model, 'predict', side_effect=ValueError('test failure')):
            result = self.predictor.predict_worst_case_delay('Shanghai', 'Rotterdam', 'sea')
            self.assertEqual(result['prediction_source'], 'OPERATIONAL_PRIOR')
            self.assertGreater(result['final_delay_presented'], 0)

    def test_canonical_suez_demo_and_audit(self):
        route = RouteRecommender(None, self.predictor, None, ScenarioManager(), demo_mode=True)
        normal = route.recommend('Shanghai', 'Rotterdam', transport_preference='sea')['recommendations'][0]
        blocked = route.recommend('Shanghai', 'Rotterdam', transport_preference='sea', scenario='SUEZ_BLOCK')['recommendations'][0]
        self.assertIn('CHOKE-SUEZ', [l['to'] for l in normal['legs']])
        self.assertNotIn('CHOKE-SUEZ', [l['to'] for l in blocked['legs']])
        self.assertIn('Port of Durban', [l['to_name'] for l in blocked['legs']])
        self.assertGreater(blocked['adjusted_eta'], normal['adjusted_eta'])
        for result in [normal, blocked]:
            self.assertEqual(round(sum(result['audit_trace']['eta'].values()), 1), result['adjusted_eta'])
            self.assertEqual(round(sum(result['audit_trace']['cost'].values()), 2), result['total_cost'])


if __name__ == '__main__': unittest.main()
