"""
Test script for the Attrition Prediction API
Tests both the old (preprocessed) and new (raw data) endpoints
"""

import requests
import json

# API Base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_feature_info():
    """Test the feature info endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Feature Info")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/feature-info")
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print(f"Total features after preprocessing: {data['total_features_after_preprocessing']}")
    print(f"\nPreprocessing steps:")
    for step in data['preprocessing_steps']:
        print(f"  {step}")
    
    return response.status_code == 200

def test_raw_data_prediction():
    """Test prediction with raw employee data"""
    print("\n" + "="*80)
    print("TEST 3: Prediction with Raw Data")
    print("="*80)
    
    # Raw employee data as it would come from database
    employee_data = {
        "Age": 41,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1102,
        "Department": "Sales",
        "DistanceFromHome": 1,
        "Education": 2,
        "EducationField": "Life Sciences",
        "EmployeeNumber": 1,
        "EnvironmentSatisfaction": 2,
        "Gender": "Female",
        "HourlyRate": 94,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5993,
        "MonthlyRate": 19479,
        "NumCompaniesWorked": 8,
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 0,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 6,
        "YearsInCurrentRole": 4,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 5
    }
    
    print(f"\nSending raw employee data (Employee #{employee_data['EmployeeNumber']})...")
    print(f"Number of fields: {len(employee_data)}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=employee_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction successful!")
        print(f"  Employee Number: {result['employee_number']}")
        print(f"  Attrition Prediction: {result['attrition_prediction']} ({'Will Leave' if result['attrition_prediction'] == 1 else 'Will Stay'})")
        print(f"  Attrition Probability: {result['attrition_probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Message: {result['message']}")
        return True
    else:
        print(f"\n✗ Prediction failed!")
        print(f"Error: {response.text}")
        return False

def test_multiple_employees():
    """Test predictions for multiple employees"""
    print("\n" + "="*80)
    print("TEST 4: Multiple Employee Predictions")
    print("="*80)
    
    employees = [
        {
            "Age": 35,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 800,
            "Department": "Research & Development",
            "DistanceFromHome": 10,
            "Education": 3,
            "EducationField": "Medical",
            "EmployeeNumber": 2,
            "EnvironmentSatisfaction": 4,
            "Gender": "Male",
            "HourlyRate": 85,
            "JobInvolvement": 4,
            "JobLevel": 3,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 8500,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": 15,
            "PerformanceRating": 4,
            "RelationshipSatisfaction": 4,
            "StockOptionLevel": 2,
            "TotalWorkingYears": 12,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 8,
            "YearsInCurrentRole": 5,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 4
        },
        {
            "Age": 28,
            "BusinessTravel": "Non-Travel",
            "DailyRate": 600,
            "Department": "Human Resources",
            "DistanceFromHome": 2,
            "Education": 1,
            "EducationField": "Human Resources",
            "EmployeeNumber": 3,
            "EnvironmentSatisfaction": 1,
            "Gender": "Female",
            "HourlyRate": 50,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Human Resources",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 3000,
            "MonthlyRate": 8000,
            "NumCompaniesWorked": 6,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 6,
            "TrainingTimesLastYear": 1,
            "WorkLifeBalance": 2,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0
        }
    ]
    
    results = []
    for emp in employees:
        response = requests.post(f"{BASE_URL}/predict", json=emp)
        if response.status_code == 200:
            results.append(response.json())
    
    print(f"\nTested {len(employees)} employees:")
    print(f"\n{'Emp #':<8} {'Prediction':<12} {'Probability':<12} {'Risk Level':<12}")
    print("-" * 50)
    for result in results:
        pred_text = "Will Leave" if result['attrition_prediction'] == 1 else "Will Stay"
        print(f"{result['employee_number']:<8} {pred_text:<12} {result['attrition_probability']:<12.2%} {result['risk_level']:<12}")
    
    return len(results) == len(employees)

def test_database_simulation():
    """Simulate fetching from database and making prediction"""
    print("\n" + "="*80)
    print("TEST 5: Database Simulation")
    print("="*80)
    
    print("\nSimulating PostgreSQL query:")
    print("SELECT * FROM employees WHERE employee_id = 12345;")
    
    # This would be your actual database query result
    db_row = {
        "Age": 42,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1200,
        "Department": "Sales",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeNumber": 12345,
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 95,
        "JobInvolvement": 3,
        "JobLevel": 3,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 9500,
        "MonthlyRate": 18000,
        "NumCompaniesWorked": 3,
        "OverTime": "No",
        "PercentSalaryHike": 13,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 15,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 10,
        "YearsInCurrentRole": 7,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 7
    }
    
    print(f"✓ Retrieved employee data from database")
    print(f"  Sending to API for prediction...")
    
    response = requests.post(f"{BASE_URL}/predict", json=db_row)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Prediction completed!")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Probability: {result['attrition_probability']:.2%}")
        return True
    else:
        print(f"\n✗ Prediction failed: {response.text}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("ATTRITION API - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Health Check", test_health_check),
        ("Feature Info", test_feature_info),
        ("Raw Data Prediction", test_raw_data_prediction),
        ("Multiple Employees", test_multiple_employees),
        ("Database Simulation", test_database_simulation)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is ready for production.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    # Make sure your FastAPI server is running before executing this script
    # Run with: python test_api.py
    
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API")
        print("Make sure your FastAPI server is running:")
        print("  uvicorn main:app --reload")
