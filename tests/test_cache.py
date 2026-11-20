import csv
import hashlib
import json

import numpy as np
import pytest

from fltoken.cache import load_update_rounds


def corpus(tmp_path):
    folder = tmp_path/"run"
    folder.mkdir()
    cache = tmp_path/"cache"/folder.name
    cache.mkdir(parents=True)
    x = np.arange(15,dtype=np.float64).reshape(3,5)
    ids = np.array([5,2,4])
    digest = hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()
    with (folder/"per_round.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["round","client_id","update_set_sha256"])
        writer.writeheader()
        writer.writerows(dict(round=1,client_id=i,update_set_sha256=digest) for i in ids)
    (folder/"round_metrics.csv").write_text("round,accuracy\n1,0.5\n",encoding="utf-8")
    (folder/"ledger.json").write_text("{}",encoding="utf-8")
    (folder/"metadata.json").write_text(json.dumps(dict(status="complete",config=dict(rounds=1,clients_per_round=3))),encoding="utf-8")
    hashes={p:hashlib.sha256((folder/p).read_bytes()).hexdigest() for p in ["per_round.csv","round_metrics.csv","ledger.json"]}
    (folder/"hashes.json").write_text(json.dumps(hashes),encoding="utf-8")
    np.savez(cache/"round-001.npz",client_ids=ids,deltas=x)
    return folder,cache.parent,x,ids


def test_common_update_cache_is_bound_to_ids_and_log(tmp_path):
    folder,cache,x,ids=corpus(tmp_path)
    [(round, loaded_ids, loaded_x, rows)]=list(load_update_rounds(folder,cache))
    assert round==1 and len(rows)==3
    np.testing.assert_array_equal(loaded_ids,ids)
    np.testing.assert_array_equal(loaded_x,x)
    assert not loaded_x.flags.writeable


@pytest.mark.parametrize("corruption",["ids","matrix","log"])
def test_cache_rejects_changed_evidence(tmp_path,corruption):
    folder,cache,x,ids=corpus(tmp_path)
    if corruption=="ids":
        ids=ids[::-1]
    elif corruption=="matrix":
        x[0,0]+=1
    else:
        (folder/"round_metrics.csv").write_text("changed",encoding="utf-8")
    np.savez(cache/folder.name/"round-001.npz",client_ids=ids,deltas=x)
    with pytest.raises(ValueError):
        list(load_update_rounds(folder,cache))
