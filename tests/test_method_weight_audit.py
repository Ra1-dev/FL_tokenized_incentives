"""Recorded-weight diagnostics distinguish underflow from legitimate empty rounds."""
import csv
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def modules():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    audit = importlib.import_module('audit_method_weights')
    method = audit.recorded_method()
    original,_ = audit.historical_scorer(method)
    return audit,original


def trace(audit,original,qualities,params=None):
    params = params or {}
    scorers = {name:original({**params,**switches}) for name,switches in audit.ABLATIONS.items()}
    for number,values in enumerate(qualities,1):
        ids,quality = np.arange(len(values)),np.array(values)
        rows = [{} for _ in values]
        for name,scorer in scorers.items():
            output = scorer.allocate(ids,quality,number)
            for rule in [name,*(['Mstar'] if name=='Afull' else [])]:
                for field in ['scores','shares','slash','selected']:
                    suffix = {'scores':'score','shares':'share'}.get(field,field)
                    for index,row in enumerate(rows):
                        row[f'{rule}_{suffix}'] = str(output[field][index])
        yield number,ids,quality,rows


def test_erased_products_are_counted_separately_from_true_zero_rounds(modules):
    audit,original = modules
    small = np.nextafter(0.,1.)
    result = audit.audit_rounds(trace(audit,original,[[small,small],[0.,0.]]),{},original)
    assert result['rounds'] == 2 and result['participations'] == 4
    for name in ['Aema','Afull']:
        stage = result['stages'][name]
        assert stage['erased_product_rounds'] == stage['subnormal_product_rounds'] == 1
        assert stage['erased_product_participations'] == 2 and stage['empty_allocation_rounds'] == 2
        assert stage['minimum_positive_rounded_weight'] is None
    assert result['stages']['Aref']['erased_product_participations'] == 0
    assert result['stages']['Aref']['empty_allocation_rounds'] == 1


@pytest.mark.parametrize('params',[dict(beta=1.,initial_reputation=0.),dict(threshold=1.,patience=1)])
def test_zero_reputation_or_slashing_is_not_reported_as_product_erasure(modules,params):
    audit,original = modules
    small = np.nextafter(0.,1.)
    result = audit.audit_rounds(trace(audit,original,[[small,small]],params),params,original)
    assert result['stages']['Afull']['erased_product_participations'] == 0
    assert result['stages']['Afull']['empty_allocation_rounds'] == 1


def test_nonzero_subnormal_products_can_still_have_rounding_distortion(modules):
    audit,original = modules
    small = np.nextafter(0.,1.)
    result = audit.audit_rounds(trace(audit,original,[[3*small,5*small]]),{},original)
    assert result['stages']['Afull']['subnormal_product_participations'] == 2
    assert result['stages']['Afull']['erased_product_participations'] == 0


@pytest.mark.parametrize('field,value',[('Mstar_share','0.3'),('Aema_score','0.3'),('Afull_slash','yes')])
def test_audit_rejects_an_allocation_which_does_not_reconstruct(modules,field,value):
    audit,original = modules
    rounds = list(trace(audit,original,[[.5,.5]]))
    rounds[0][3][0][field] = value
    with pytest.raises(ValueError,match='recorded|original'):
        audit.audit_rounds(rounds,{},original)


@pytest.mark.parametrize('rounds',[[],[(2,np.arange(2),np.ones(2),None)]])
def test_empty_or_skipped_rounds_are_rejected(modules,rounds):
    audit,original = modules
    with pytest.raises(ValueError,match='round'):
        audit.audit_rounds(rounds,{},original)


def test_partial_diagnostic_cannot_overwrite_the_default_complete_result(modules,monkeypatch):
    import sys
    audit,_ = modules
    monkeypatch.setattr(sys,'argv',['audit_method_weights.py','--available'])
    with pytest.raises(SystemExit) as stopped:
        audit.main()
    assert stopped.value.code == 2


@pytest.mark.parametrize('mutation',['order','round','matrix','missing'])
def test_recorded_rows_require_the_exact_sample_and_coverage(modules,tmp_path,mutation):
    audit,_ = modules
    config = dict(num_clients=7,clients_per_round=3,rounds=2,seed=927)
    rows = [dict(round=number,client_id=int(client),Mstar_score='.5',update_set_sha256='common')
            for number in [1,2] for client in audit.sample_clients(7,3,number,927)]
    def write():
        with (tmp_path/'per_round.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write()
    assert len(list(audit.recorded_rounds(tmp_path,config))) == 2
    if mutation == 'order':rows[:3] = rows[:3][::-1]
    elif mutation == 'round':rows[0]['round'] = 2
    elif mutation == 'matrix':rows[0]['update_set_sha256'] = 'different'
    else:rows.pop()
    write()
    with pytest.raises(ValueError,match='weight audit'):
        list(audit.recorded_rounds(tmp_path,config))
