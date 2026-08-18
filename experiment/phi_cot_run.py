"""Phi4-mini with CHAIN-OF-THOUGHT prompt, written to a SEPARATE run dir (phi4-mini-cot).
The direct-answer phi4-mini run is kept untouched. Local (Ollama), free, resumable."""
import sys, copy, json, re as _re
from pathlib import Path
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[1]
for p in [PROJECT_ROOT, PROJECT_ROOT/'experiment']:
    sys.path.insert(0, str(p))
from openai import OpenAI
from modules.data_intake import load_data
from modules.test_runner import run_tests
from modules.eval_all import eval_all
from registry.seed_kb import KB_EXPECTED_BEHAVIOR
from run_utils import update_manifest

NAME='phi4-mini-cot'; MODEL='phi4-mini'
DATASET='gsm8k_symbolic100'; THRESHOLD=0.80; JUDGE_MODEL='gpt-4o-mini'; N_BASELINE=3; N_TEST_Q=100
OUT=PROJECT_ROOT/'experiment'/'output'
BENCH=OUT/'benchmark'/DATASET/'generated.csv'

def make_client(local=True):
    if local: return OpenAI(api_key='ollama', base_url='http://127.0.0.1:11434/v1')
    k=(PROJECT_ROOT/'doc'/'Key_o.txt').read_text().strip().strip('"').strip("'")
    return OpenAI(api_key=k)

generated_df=pd.read_csv(BENCH)
intake=load_data(source='csv_upload', csv_path=str(OUT/'run_base.csv'), input_cols=['question'],
                 output_cols=['ground_truth'], type_overrides={'ground_truth':'numeric'},
                 row_mode='all', acceptance_threshold=THRESHOLD)
print(f'{NAME}: benchmark {len(generated_df)} rows | {generated_df["Source Row"].nunique()} questions', flush=True)

# CHAIN-OF-THOUGHT prompt (parser already handles the #### fallback)
MATH_PROMPT=('You are a math problem solver. Solve the grade-school math problem step by step. '
             'Then give the final answer on its own line in exactly this format:\n#### <number>\n'
             '(a plain number only — no units, $, commas, or words).')
KB_PROMPT=('Solve the grade-school math problem step by step. '
           'If the question has a false premise, an impossible or fictional scenario, made-up '
           'entities, or is missing information needed to solve it, do NOT invent an answer — '
           'briefly explain why it cannot be answered. Otherwise end with the final answer on '
           'its own line as a plain number only (no units, $, words, or commas):\n#### <number>')
def parse_math(raw):
    raw=raw.strip()
    for cand in [raw]+_re.findall(r'\{[^{}]*\}', raw):
        try:
            v=json.loads(cand)
            if isinstance(v,dict) and ('ground_truth' in v or 'answer' in v):
                return str(v.get('ground_truth', v.get('answer','')))
        except Exception: pass
    m=_re.findall(r'####\s*\$?(-?[\d,]+\.?\d*)', raw)
    if m: return m[-1].replace(',','')
    nums=_re.findall(r'-?[\d,]+\.?\d*', raw)
    return nums[-1].replace(',','') if nums else raw
def make_sut(client, model, prompt):
    RAW=[]
    def sut(fields):
        user='\n'.join(f'{k.upper()}: {v}' for k,v in fields.items())
        try:
            r=client.chat.completions.create(model=model, temperature=0.0, max_tokens=900,
                messages=[{'role':'system','content':prompt},{'role':'user','content':user}])
            raw=(r.choices[0].message.content or '').strip()
        except Exception as _e:
            RAW.append(f'[ERROR] {_e}'); raise
        RAW.append(raw); return {'ground_truth': parse_math(raw)}
    return sut, RAW

client=make_client(local=True)
RUN_DIR=OUT/'runs'/f'{NAME}__{DATASET}'; RUN_DIR.mkdir(parents=True, exist_ok=True)
update_manifest(RUN_DIR, sut_model=NAME, provider='local', dataset=DATASET, threshold=THRESHOLD)
resp_path=RUN_DIR/'responses.csv'
if resp_path.exists():
    responses=pd.read_csv(resp_path); done=set(responses['Source Row'].unique())
else:
    responses=pd.DataFrame(); done=set()
all_q=sorted(generated_df['Source Row'].unique())[:N_TEST_Q]
todo=[r for r in all_q if r not in done]
print(f'{NAME}: {len(done)} done, {len(todo)} to run', flush=True)
for n,row_idx in enumerate(todo,1):
    sub=generated_df[generated_df['Source Row']==row_idx]
    ri=copy.copy(intake); ri.df=intake.df.iloc[[row_idx]].reset_index(drop=True)
    sub_kb=sub[sub['Category']=='Knowledge Boundary']; sub_main=sub[sub['Category']!='Knowledge Boundary']
    parts=[]
    sut,RAW=make_sut(client, MODEL, MATH_PROMPT)
    rm=run_tests(generated_df=sub_main, sut=sut, intake=ri, gate_only=True, n_baseline=N_BASELINE)
    rm['Raw Response']=(list(RAW)+['']*len(rm))[:len(rm)]; parts.append(rm)
    if len(sub_kb):
        sutk,RAWk=make_sut(client, MODEL, KB_PROMPT)
        rk=run_tests(generated_df=sub_kb, sut=sutk, intake=ri, gate_only=True, n_baseline=0)
        rk['Raw Response']=(list(RAWk)+['']*len(rk))[:len(rk)]; parts.append(rk)
    res=pd.concat(parts, ignore_index=True); res['Source Row']=row_idx
    responses=pd.concat([responses,res], ignore_index=True); responses.to_csv(resp_path, index=False)
    print(f'[{len(done)+n}/{len(all_q)}] q{row_idx} (+{len(res)}; total {len(responses)})', flush=True)
update_manifest(RUN_DIR, n_responses=int(len(responses)))
print(f'{NAME} RESPONSES DONE: {len(responses)} rows', flush=True)

def acc_table(eval_df):
    TAG={'Semantic Variation':'SV','Input Quality':'IQ','Structural/Format':'F','Context Interference':'CI',
         'Context Load':'CL','Conflict Instruction Stress':'CIS','Knowledge Boundary':'KB'}
    eval_df=eval_df.copy(); eval_df['fam']=eval_df['Category'].map(TAG)
    eval_df['correct']=pd.to_numeric(eval_df['Sim: outputs'],errors='coerce')>=THRESHOLD
    base=eval_df[eval_df['Category']=='Temporal Consistency']; ab=base['correct'].mean() if len(base) else 1.0
    tau=0.80*ab; st=eval_df[eval_df['fam'].notna()]
    tbl=st.pivot_table(index='fam',columns='Classified Level',values='correct',aggfunc='mean'); tbl.insert(0,'L0',ab)
    order=['L1','L2','L3','L4','L5']
    cp={f: next((L for L in order if L in r.index and pd.notna(r[L]) and r[L]<tau), None) for f,r in tbl.iterrows()}
    return tbl, tau, cp
overrides={'Knowledge Boundary':{'method':'llm_judge','expected_behavior':KB_EXPECTED_BEHAVIOR,'judge_model':JUDGE_MODEL}}
edf=eval_all(results_df=pd.read_csv(resp_path), intake=intake, overrides=overrides, client=make_client(local=False))
edf.to_csv(RUN_DIR/'eval.csv', index=False)
tbl,tau,cp=acc_table(edf); tbl.round(3).to_csv(RUN_DIR/'metrics_acc.csv')
pd.DataFrame({'family':list(cp),'CP':list(cp.values())}).to_csv(RUN_DIR/'metrics_summary.csv',index=False)
print(f'{NAME} EVAL DONE: baseline acc(L0)={tau/0.80:.3f}  CP={cp}', flush=True)
print(tbl.round(3).to_string(), flush=True)
