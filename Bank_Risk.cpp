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