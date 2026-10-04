import sys
sys.path.insert(0, '.')

import fitz
from fastapi import FastAPI
from fastapi.testclient import TestClient
import part5


def make_pdf(text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data

# 1. Pure PyMuPDF extraction
pdf = make_pdf('Aisha Khan\naisha@example.com\nPython SQL FastAPI\nBS Information Technology')
ex = part5.extract_pdf(pdf)
assert ex['page_count'] == 1
assert 'Aisha Khan' in ex['text']
assert 'Python SQL FastAPI' in ex['text']

# 2. AI parser mocked so no network/API dependency is needed for test
part5.groq_call = lambda *a, **k: '{"document_type":"CV","summary":"IT CV","person":{"name":"Aisha Khan","email":"aisha@example.com","phone":"","location":""},"education":["BS Information Technology"],"experience":[],"skills":["Python","SQL","FastAPI"],"certifications":[],"financial_information":[],"important_dates":[],"missing_or_unclear_information":[],"verification_required":[]}'
analysis = part5.analyze_with_ai('cv.pdf', ex['text'])
assert analysis['document_type'] == 'CV'
assert analysis['person']['email'] == 'aisha@example.com'

# 3. Finance
f = part5.analyze_finance(part5.FinanceRequest(budget=5000, tasks=[
    part5.FinanceTask(title='Required fee', cost=4000),
    part5.FinanceTask(title='Optional course', cost=3000, optional=True),
]))
assert f['over_budget'] is True
assert any(x['action']=='skip_optional' for x in f['recommendations'])

# 4. Verification
v = part5.verification_check(part5.VerificationRequest(items=[
    part5.VerificationItem(field='email', value='bad-email'),
    part5.VerificationItem(field='name', value='Aisha'),
    part5.VerificationItem(field='CNIC', value=None),
]))
assert v['overall'] == 'REVIEW'
assert v['review_count'] == 2

# 5. Adaptive planning
p = part5.adaptive_plan(part5.AdaptivePlanRequest(
    previous_days=30, days=10, previous_budget=10000, budget=5000,
    tasks=[
        part5.AdaptiveTask(id='1', title='Required', due=10, off=.5, cost=4000),
        part5.AdaptiveTask(id='2', title='Optional', due=20, off=.8, cost=3000, optional=True),
    ]
))
assert p['tasks'][0]['due'] == 5
assert p['tasks'][1]['skipped'] is True
assert p['planned_cost'] == 4000
assert len(p['changes']) >= 2

# 6. FastAPI endpoints
app = FastAPI()
app.include_router(part5.router, prefix='/part5')
client = TestClient(app)

r = client.post('/part5/document/analyze', files={'file': ('cv.pdf', pdf, 'application/pdf')})
assert r.status_code == 200, r.text
assert r.json()['status'] in ('SUCCESS','PARTIAL')

r = client.post('/part5/finance/analyze', json={'budget': 5000, 'tasks':[{'title':'A','cost':6000}]})
assert r.status_code == 200, r.text
assert r.json()['over_budget'] is True

r = client.post('/part5/verification/check', json={'items':[{'field':'email','value':'a@b.com'}]})
assert r.status_code == 200, r.text
assert r.json()['overall'] == 'PASS'

r = client.post('/part5/adaptive-plan', json={
    'previous_days':30,'days':15,'previous_budget':10000,'budget':5000,
    'tasks':[{'id':'1','title':'A','due':10,'off':0.5,'cost':6000}]
})
assert r.status_code == 200, r.text
assert r.json()['tasks'][0]['due'] == 8

print('ALL PART 5 TESTS PASSED')
