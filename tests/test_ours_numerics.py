"""Finite-range geometry, cached quality, and penalty-boundary contracts."""
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from fltoken.ours import OursScorer


@pytest.mark.parametrize('multiple',[1,2,3,16])
def test_orthogonality_is_not_a_positive_subnormal_residue(multiple):
    small=multiple*np.nextafter(0.,1.)
    matrix=np.array([[1.,small,small],[-2*small,1.,1.]])
    assert sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(*matrix)),Fraction(0))==0
    result=OursScorer(dict(patience=1)).score([0,1],matrix,1)
    np.testing.assert_array_equal(result['scores'],[0.,0.])
    np.testing.assert_array_equal(result['slash'],[True,True])


@pytest.mark.parametrize('count',[2,3,7,9,10,13])
@pytest.mark.parametrize('reference',[True,False])
def test_aligned_smallest_subnormal_rewards(count,reference):
    matrix=np.full((count,3),np.nextafter(0.,1.))
    result=OursScorer(dict(reference=reference)).score(range(count),matrix,1)
    np.testing.assert_allclose(result['scores'],np.ones(count),atol=2e-15,rtol=2e-14)
    np.testing.assert_allclose(result['shares'],np.full(count,1/count),rtol=0,atol=2e-16)
    assert not result['slash'].any()


def test_all_normal_inputs_with_unrepresentable_reference_midpoint():
    tiny=np.finfo(float).tiny
    matrix=np.array([[-tiny],[tiny],[-np.nextafter(tiny,np.inf)]])
    assert np.all(np.abs(matrix)>=tiny)
    enlarged=np.ldexp(matrix,1022)
    np.testing.assert_array_equal(np.ldexp(enlarged,-1022),matrix)
    for values in [matrix,enlarged]:
        result=OursScorer().score(range(3),values,1)
        np.testing.assert_array_equal(result['scores'],[1.,0.,0.])


@pytest.mark.parametrize('vertical',[np.nextafter(2.,0.),2.,np.nextafter(2.,np.inf)])
def test_cosine_rounding_preserves_zero_penalty_boundary(vertical):
    small=np.nextafter(0.,1.)
    matrix=np.array([[small,vertical],[2.,-vertical]])
    square=Fraction(float(small))**2/(Fraction(float(small))**2+Fraction(float(vertical))**2)
    with localcontext() as context:
        context.prec=1200
        expected=float((Decimal(square.numerator)/Decimal(square.denominator)).sqrt())
    result=OursScorer(dict(reference=False,patience=1)).score(range(2),matrix,1)
    assert result['scores'][0]==expected
    assert bool(result['slash'][0])==(expected==0.)


@pytest.mark.parametrize('beta',[0.,.5,.9])
@pytest.mark.parametrize('penalties',[True,False])
def test_geometry_cache_is_independent_of_economic_parameters(beta,penalties):
    matrix=np.array([[0.,0.,0.,0.,0.],[-3.,0.,6.,10.,5.],[-3.,0.,6.,10.,5.]])
    quality=OursScorer(dict(ema=False,penalties=False)).score(range(3),matrix,1)['scores']
    settings=dict(threshold=float(quality[-1]),patience=1,beta=beta,penalties=penalties)
    direct=OursScorer(settings).score(range(3),matrix,1)
    cached=OursScorer(settings).allocate(range(3),quality,1)
    for field in ['scores','shares','slash','selected']:
        np.testing.assert_array_equal(direct[field],cached[field])
    assert direct['state']==cached['state']


@pytest.mark.parametrize('count',[3,7,9,10])
@pytest.mark.parametrize('reference',[True,False])
def test_exact_rescaling_keeps_threshold_decisions_and_permuted_state(count,reference):
    rng=np.random.default_rng(914718)
    matrix=rng.integers(-12,13,(count,5)).astype(float)
    matrix[0]=0
    matrix[1]=matrix[2]
    canonical=OursScorer(dict(reference=reference)).score(range(count),matrix,1)['scores']
    for threshold in [0.,.2,.5,float(canonical[-1])]:
        for power in [-1074,-1022,1018]:
            scaled=np.ldexp(matrix,power)
            assert all(Fraction(float(b))==Fraction(float(a))*Fraction(2)**power for a,b in zip(matrix.ravel(),scaled.ravel()))
            settings=dict(reference=reference,threshold=threshold,patience=2)
            a,b=OursScorer(settings),OursScorer(settings)
            order=rng.permutation(count)
            inverse=np.argsort(order)
            for number in [1,2]:
                left=a.score(range(count),matrix,number)
                right=b.score(order,scaled[order],number)
                for field in ['scores','shares','slash','selected']:
                    np.testing.assert_array_equal(left[field],right[field][inverse])
                assert left['state']==right['state']


def exact_qualities(matrix, excluding):
    """Independent rational coordinates and high-precision decimal square root."""
    points = [[Fraction(float(value)) for value in row] for row in matrix]
    result = []
    for index,row in enumerate(points):
        others = [point for position,point in enumerate(points) if not excluding or position != index]
        if not others:
            result.append(0.)
            continue
        columns = [sorted(column) for column in zip(*others)]
        count = len(others)
        reference = ([column[count//2] if count%2 else (column[count//2-1]+column[count//2])/2
                      for column in columns] if excluding else [sum(column,Fraction(0))/count for column in columns])
        dot = sum((a*b for a,b in zip(row,reference)),Fraction(0))
        if dot <= 0:
            result.append(0.)
            continue
        norm_product = sum((a*a for a in row),Fraction(0))*sum((b*b for b in reference),Fraction(0))
        square = dot*dot/norm_product
        with localcontext() as context:
            context.prec = 120
            result.append(float((Decimal(square.numerator)/Decimal(square.denominator)).sqrt()))
    return np.array(result)


@pytest.mark.parametrize('count',[1,2,3,4,7,9,10,13])
@pytest.mark.parametrize('reference',[True,False])
def test_mixed_exponents_match_independent_geometry(count,reference):
    rng = np.random.default_rng(651791+count)
    exponents = [-1074,-1070,-1022,-600,-3,0,512,1018]
    for _ in range(10):
        matrix = np.ldexp(rng.integers(-7,8,(count,5)).astype(float),rng.choice(exponents,(count,5)))
        matrix.flags.writeable = False
        before = matrix.copy()
        expected = exact_qualities(matrix,reference)
        actual = OursScorer(dict(reference=reference,patience=1)).score(range(count),matrix,1)
        np.testing.assert_allclose(actual['scores'],expected,atol=2e-15,rtol=2e-13)
        np.testing.assert_array_equal(actual['scores']==0,expected==0)
        np.testing.assert_array_equal(actual['slash'],expected==0)
        np.testing.assert_array_equal(matrix,before)
