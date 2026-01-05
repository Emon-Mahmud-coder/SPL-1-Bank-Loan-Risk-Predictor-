#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <random>
#include <chrono>

using namespace std;


// DATA STRUCTURES


struct LoanRecord {
    int age;
    double income;
    int loanAmount;
    int creditScore;
    int monthsEmployed;
    int numCreditLines;
    double interestRate;
    int loanTerm;
    double dtiRatio;
    int education;           
    int employmentType;      
    int maritalStatus;       
    int hasMortgage;         
    int hasDependents;       
    int loanPurpose;         
    int hasCoSigner;         
    int defaultRisk;         
};

struct DecisionNode {
    bool isLeaf;
    int predictedClass;      
    int featureIndex;        
    double threshold;        
    int leftChild;           
    int rightChild;          
};

struct DecisionTree {
    vector<DecisionNode> nodes;
    int rootIndex;
};

struct RandomForest {
    vector<DecisionTree> trees;
    int numTrees;
};

struct RiskAnalysis {
    int predictedClass;
    double confidence;
    vector<string> riskFactors;
    vector<string> positiveFactors;
    string riskLevel;
    vector<string> recommendations;
};

const vector<string> FEATURE_NAMES = {
    "Age", "Income", "Loan Amount", "Credit Score", 
    "Months Employed", "Credit Lines", "Interest Rate", "Loan Term",
    "DTI Ratio", "Education", "Employment Type", "Marital Status",
    "Has Mortgage", "Has Dependents", "Loan Purpose", "Has Co-Signer"
};


// Split a CSV line into tokens
vector<string> splitCSV(const string& line) {
    vector<string> tokens;
    stringstream ss(line);
    string token;
    
    while (getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        tokens.push_back(token);
    }
    
    return tokens;
}

// Safe string to int conversion
int safeStoi(const string& str, int defaultValue = 0) {
    try {
        if (str.empty()) return defaultValue;
        return stoi(str);
    } catch (...) {
        return defaultValue;
    }
}

// Safe string to double conversion
double safeStod(const string& str, double defaultValue = 0.0) {
    try {
        if (str.empty()) return defaultValue;
        return stod(str);
    } catch (...) {
        return defaultValue;
    }
}

// Encode categorical Education values

int encodeEducation(const string& value) {
    if (value == "High School") return 0;
    if (value == "Bachelor's" || value == "Bachelor") return 1;
    if (value == "Master's" || value == "Master") return 2;
    if (value == "PhD") return 3;
    return 0; // default
}

// Decode education code to name

string getEducationName(int code) {
    switch(code) {
        case 0: return "High School";
        case 1: return "Bachelor's";
        case 2: return "Master's";
        case 3: return "PhD";
        default: return "Unknown";
    }
}

// Encode categorical EmploymentType values
int encodeEmploymentType(const string& value) {
    if (value == "Unemployed") return 0;
    if (value == "Self-employed") return 1;
    if (value == "Full-time") return 2;
    if (value == "Part-time") return 3;
    return 2; // default to Full-time
}

string getEmploymentName(int code) {
    switch(code) {
        case 0: return "Unemployed";
        case 1: return "Self-employed";
        case 2: return "Full-time";
        case 3: return "Part-time";
        default: return "Unknown";
    }
}

// Encode categorical MaritalStatus values
int encodeMaritalStatus(const string& value) {
    if (value == "Single") return 0;
    if (value == "Married") return 1;
    if (value == "Divorced") return 2;
    return 0; // default
}

// Encode categorical LoanPurpose values
int encodeLoanPurpose(const string& value) {
    if (value == "Auto") return 0;
    if (value == "Business") return 1;
    if (value == "Education") return 2;
    if (value == "Home") return 3;
    if (value == "Other") return 4;
    return 4; // default
}

// Decode loan purpose
string getLoanPurposeName(int code) {
    switch(code) {
        case 0: return "Auto";
        case 1: return "Business";
        case 2: return "Education";
        case 3: return "Home";
        case 4: return "Other";
        default: return "Unknown";
    }
}

// Encode Yes/No to 1/0
int encodeYesNo(const string& value) {
    if (value == "Yes" || value == "yes" || value == "YES" || value == "1") return 1;
    return 0;
}




int main(){

    //Initialize random  seed

    srand(time(0));
      
    cout << "=========================================" << endl;
    cout << "  LOAN RISK PREDICTION SYSTEM" << endl;
    cout << "  Using Random Forest Algorithm" << endl;
    cout << "  Advanced Analytics Edition" << endl;
    cout << "=========================================" << endl;
    
    //Step-01: Data Preprocessing

    string datasetFile = "data.csv";


}



