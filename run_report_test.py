from src.report_generator import ReportGenerator
from langchain.schema import Document

print('RUN_REPORT_TEST: start')
rg = ReportGenerator()
print('RUN_REPORT_TEST: instantiated ReportGenerator')
# If vectorstore empty, add sample documents
if rg.vs.is_empty():
    sample_texts = [
        "Community-acquired pneumonia (CAP) in adults is commonly caused by Streptococcus pneumoniae.",
    ]
    docs = [Document(page_content=t, metadata={'source':f'sample{i}'}) for i,t in enumerate(sample_texts,1)]
    rg.vs.add_documents(docs)
    print('RUN_REPORT_TEST: added sample documents to vectorstore')

# Change to a different patient whose symptoms match the seeded pneumonia sample
patient = {'Name':'Aisha Khan','Age':'45','Symptoms':'inflammation in  stomach , liver swelling, constipation, vomit ','Duration':'3 days'}
question = 'Given the patient data (45-year-old with stomach pain, vomitingcd "c:\ai project\MedRAG"; python "c:\ai project\MedRAG\run_report_test.py"), what is the most likely diagnosis, recommended diagnostic tests, and empiric antibiotic management? Please produce a structured clinical report.'
try:
    report, retrieved = rg.generate(patient, question, top_k=3, use_web=True, structured=True)
    print('RUN_REPORT_TEST: report generated')
    print('--- REPORT START ---')
    print(report)
except Exception as e:
    print('RUN_REPORT_TEST: exception during generate:', repr(e))
    raise
print('\n--- Retrieved docs:')
for d in retrieved:
    print('-', getattr(d, 'metadata', {}).get('source','?'), '->', (getattr(d,'page_content','')[:120]).replace('\n',' '))
