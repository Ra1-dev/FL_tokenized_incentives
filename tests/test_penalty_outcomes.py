"""Slashing costs must count repeated events without duplicating clients."""
import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def accounting(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("summarize_grid_settlement")


def history(module):
    scale=10**18
    ledger=[]
    oracle=module.ChainLedger(range(2),3)
    states=[]
    for number in range(1,5):
        shares=[scale//2]*2 if number==1 else [0,0]
        item=dict(round=number,ids=[0,1],shares1e18=shares,payouts=[1000*s for s in shares],
                  slash=[number>1,False],remainder=1000*scale-1000*sum(shares))
        ledger.append(item)
        transition=oracle.settle(item)
        for client in oracle.accounts:
            oracle.claim(client)
        states.append(dict(**transition,state=oracle.snapshot()))
    return ledger,states


def test_repeat_slashes_burn_only_pending_rewards_and_net_can_be_negative(accounting):
    ledger,states=history(accounting)
    result=accounting.economic_outcomes(ledger,states,{1:"zero"})
    honest=result["groups"]["honest"]
    assert honest["slash_events"]==3 and honest["ever_slashed_clients"]==1
    assert honest["clients_with_balance_loss"]==1
    assert honest["burned_rewards"]==95*10**18  # round-four maturity is protected
    assert honest["burned_stake"]==271*10**17
    assert honest["net_token_outcome"]==3779*10**17
    assert result["groups"]["adversary"]["net_token_outcome"]==500*10**18
    empty=dict(round=1,ids=[0,1],shares1e18=[0,0],payouts=[0,0],slash=[True,False],remainder=1000*10**18)
    oracle=accounting.ChainLedger(range(2),3)
    transition=oracle.settle(empty)
    outcome=accounting.economic_outcomes([empty],[dict(**transition,state=oracle.snapshot())],{1:"zero"})
    assert outcome["groups"]["honest"]["net_token_outcome"]==-10*10**18
    assert outcome["minted_attacker_share"] is None


def test_identical_allocations_can_have_different_settlement_costs(accounting):
    ledger,states=history(accounting)
    original=accounting.economic_outcomes(ledger,states,{1:"zero"})
    no_penalty=copy.deepcopy(ledger)
    for row in no_penalty:
        row["slash"]=[False,False]
    result=accounting.penalty_ablation(dict(Mstar=ledger,Afull=ledger,Aema=no_penalty),original,2,3,{1:"zero"})
    assert result["allocations_identical"]
    off=result["stages"]["Aema"]["groups"]["honest"]
    on=result["stages"]["Afull"]["groups"]["honest"]
    assert off["minted_rewards"]==on["minted_rewards"]==500*10**18
    assert off["net_token_outcome"]-on["net_token_outcome"]==1221*10**17
    assert off["ever_slashed_clients"]==off["burned_rewards"]==off["burned_stake"]==0
    aggregate=accounting.aggregate_components([dict(horizon="main",**result)]*2)
    assert aggregate["main"]["groups"]["Afull"]["honest"]["ever_slashed_clients"]==2


@pytest.mark.parametrize("change",["ema_flag","different_full","wrong_chain"])
def test_component_comparison_rejects_changed_inputs_or_chain_balance(accounting,change):
    ledger,states=history(accounting)
    original=accounting.economic_outcomes(ledger,states,{1:"zero"})
    no_penalty=copy.deepcopy(ledger)
    for row in no_penalty:row["slash"]=[False,False]
    ledgers=dict(Mstar=ledger,Afull=copy.deepcopy(ledger),Aema=no_penalty)
    if change=="ema_flag":ledgers["Aema"][0]["slash"][0]=True
    elif change=="different_full":ledgers["Afull"][0]["payouts"][0]+=1
    else:original["groups"]["honest"]["burned_stake"]+=1
    with pytest.raises(ValueError):
        accounting.penalty_ablation(ledgers,original,2,3,{1:"zero"})


@pytest.mark.parametrize("gaps", [False, True])
def test_zero_threshold_patience_equal_to_vesting_protects_prior_rewards(accounting,gaps):
    from fltoken.metrics import payout, ROUND_POOL
    from fltoken.ours import OursScorer
    scorer=OursScorer(dict(threshold=0.,patience=3))
    oracle=accounting.ChainLedger(range(2),3)
    ledger,states=[],[]
    bad_rounds={2,4,6} if gaps else {2,3,4}
    for number in range(1,max(bad_rounds)+1):
        ids=[0,1] if number==1 or number in bad_rounds else [1]
        quality=[1.,1.] if number==1 else [0.,1.] if number in bad_rounds else [1.]
        result=scorer.allocate(ids,quality,number)
        units,amounts=payout(result["shares"])
        item=dict(round=number,ids=ids,shares1e18=units,payouts=amounts,
                  slash=result["slash"].tolist(),remainder=ROUND_POOL-sum(amounts))
        ledger.append(item)
        transition=oracle.settle(item)
        for client in oracle.accounts:oracle.claim(client)
        states.append(dict(**transition,state=oracle.snapshot()))
    honest=accounting.economic_outcomes(ledger,states,{1:"zero"})["groups"]["honest"]
    assert honest["slash_events"]==honest["ever_slashed_clients"]==1
    assert honest["burned_stake"]==10*10**18
    assert honest["burned_rewards"]==0
    assert honest["retained_rewards"]==500*10**18
    assert honest["net_token_outcome"]==490*10**18
