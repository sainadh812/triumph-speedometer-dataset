"""Pure loss aggregation and mock evaluation tests; no neural execution."""
import unittest
import numpy as np

from test_sampling import isolate, MockModel, batch, tensor


class EvaluationDenominatorTests(unittest.TestCase):
    def test_visible_points_not_batches_determine_heatmap_average(self):
        space=isolate();accumulate=space['_accumulate_evaluation_components'];sums,weights={},{}
        accumulate(sums,weights,{'heatmap':1.,'visibility':0.,'total':1.},1,1,1)
        accumulate(sums,weights,{'heatmap':9.,'visibility':0.,'total':9.},1,9,9)
        result=space['_finalize_evaluation_losses'](sums,weights)
        self.assertAlmostEqual(result['heatmap'],8.2)
        self.assertAlmostEqual(result['total'],8.2)
        self.assertEqual(weights['heatmap'],10.)

    def test_localizer_terms_are_partition_invariant_and_keep_loss_coefficients(self):
        space=isolate();accumulate=space['_accumulate_evaluation_components']
        visible=np.array([1,0,5,2]);known=np.array([2,0,8,4]);present=np.array([1,0,1,1])
        means={'heatmap':np.array([1.,0.,3.,8.]),'visibility':np.array([2.,0.,4.,6.]),
               'bbox':np.array([.1,0.,.2,.8]),'bbox_heatmap':np.array([5.,0.,9.,11.]),
               'mask_bce':np.array([.1,.4,.8,.2]),'mask_dice':np.array([.3,.1,.7,.5]),
               'presence':np.array([.2,.6,.9,.3])}
        den={'heatmap':visible,'visibility':known,'bbox':present,'bbox_heatmap':present,
             'mask_bce':np.ones(4),'mask_dice':np.ones(4),'presence':np.ones(4)}
        coefficients={'heatmap':1.,'visibility':.25,'bbox':5.,'bbox_heatmap':1.,
                      'mask_bce':1.,'mask_dice':1.,'presence':.5}
        def finish(partitions):
            sums,weights={},{}
            for indices in partitions:
                c={key:float(np.sum(value[indices]*den[key][indices])/max(1.,np.sum(den[key][indices])))
                   for key,value in means.items()}
                c['total']=sum(coefficients[key]*value for key,value in c.items())
                accumulate(sums,weights,c,len(indices),int(known[indices].sum()),
                           int(visible[indices].sum()),int(present[indices].sum()))
            return space['_finalize_evaluation_losses'](sums,weights),weights
        expected,denominators=finish([[0,1,2,3]])
        for partitions in [[[0],[1],[2],[3]],[[0,1],[2,3]],[[0],[1,2,3]]]:
            actual,weights=finish(partitions)
            for key in expected:self.assertAlmostEqual(actual[key],expected[key])
            self.assertEqual(weights,denominators)
        self.assertEqual(denominators['bbox'],3.)
        self.assertEqual(denominators['heatmap'],8.)
        self.assertEqual(denominators['visibility'],14.)
        self.assertAlmostEqual(expected['total'],sum(coefficients[k]*expected[k] for k in coefficients))

    def test_evaluate_subgroups_exclude_unlabeled_rows_from_point_denominators(self):
        def fake_loss(output,current,hw,sigma):
            visible=np.asarray(current['visible']).sum(1)
            known=np.asarray(current['visibility_known']).sum(1)
            values=np.asarray(current['image'])
            heatmap=float(np.sum(values*visible)/max(1.,visible.sum()))
            visibility=float(np.sum(2*values*known)/max(1.,known.sum()))
            return heatmap+.25*visibility,{'heatmap':heatmap,'visibility':visibility,'total':heatmap+.25*visibility}
        space=isolate(training_loss=fake_loss);model=MockModel()
        raw=batch([1,9,3],['day','day','night'])
        raw['visible']=tensor([[1,0],[0,0],[1,1]])
        raw['visibility_known']=tensor([[1,1],[0,0],[1,1]])
        result=space['evaluate'](model,[raw],'unused',space['TrainConfig'](amp=False),['pivot','tip'])
        self.assertEqual(model.calls,1)
        self.assertAlmostEqual(result['loss']['heatmap'],7/3)
        self.assertAlmostEqual(result['loss']['visibility'],4.)
        self.assertAlmostEqual(result['loss']['total'],10/3)
        self.assertEqual(result['loss_denominators'],{'heatmap':3.,'visibility':4.})
        self.assertAlmostEqual(result['by_lighting']['day']['loss']['total'],1.5)
        self.assertAlmostEqual(result['by_lighting']['night']['loss']['total'],4.5)
        self.assertEqual(result['by_lighting']['day']['loss_denominators']['heatmap'],1.)
        self.assertEqual(result['by_lighting']['day']['samples'],2)

    def test_unrecognized_loss_term_requires_an_explicit_denominator(self):
        space=isolate()
        with self.assertRaisesRegex(ValueError,'undefined'):
            space['_accumulate_evaluation_components']({}, {}, {'new_term':1.},1,1,1)


if __name__=='__main__':unittest.main()
