from backend.ingest_all import FEC_PROHIBITED_DATA


def test_fec_prohibited_contains_key_categories():
    categories = {item.get('category') for item in FEC_PROHIBITED_DATA}
    expected = {
        'tax_advice', 'financial_solicitation', 'defamation', 'false_identity',
        'false_claims', 'coercion', 'medical_advice', 'foreign_national',
        'corporate_contribution', 'straw_donor', 'contribution_limits',
        'coordination', 'disclaimer_requirement', 'reporting', 'federal_contractor'
    }

    missing = expected - categories
    assert not missing, f"Missing expected FEC prohibited categories: {missing}"
