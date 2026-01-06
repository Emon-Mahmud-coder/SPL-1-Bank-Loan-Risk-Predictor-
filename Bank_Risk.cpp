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

// data Loading Preprocessing

vector <LoanRecord> loadAndPreprocessDataset( string& filename )
{

vector<LoanRecord> dataset;

ifstream file (filename);

if(!file.is_open())
{
    cerr << "Error: Could not open file" << endl;
    return dataset;
}

string line;
bool isFirstLine = true;
int lineCount = 0;
int loadedCount = 0;
int invalidCount = 0;

cout << "Loading Dataset from" << filename << "..." ;
cout << endl;

cout << "This may take a moment for large dataset...";
cout << endl;

auto startTime = chrono::high_resolution_clock::now();

while(getline(file , line))
{

    lineCount++;

    //Skip First Line 
    if(isFirstLine){
        
        continue;
        isFirstLine = true;
    }
     
    //Skip Empty Line;

    if(line.empty()){
        continue;
    }

    vector<string> tokens = splitCSV(line);

    //Ensure We Have enough columns
   if(line.size() < 18){
    invalidCount++;
    continue;
   }

  LoanRecord record;
  //Parse and Convert each field

  record.age = safeStoi(tokens[1] , 30);
  record.income = safeStod(tokens[2] , 50000.0);
  record.loanAmount = safeStoi(tokens[3] ,10000 );
  record.creditScore = safeStoi(tokens[4] , 650);
  record.monthsEmployed = safeStoi(tokens[5], 12);
  record.numCreditLines = safeStoi(tokens[6] , 2);
  record.interestRate = safeStoi(tokens[7] , 5.0 );
  record.loanTerm = safeStoi(tokens[8], 36);
  record.dtiRatio = safeStod(tokens[9], 0.3);
  record.education = encodeEducation(tokens[10]);
  record.employmentType = encodeEmploymentType(tokens[11]);
  record.maritalStatus = encodeMaritalStatus(tokens[12]);
  record.hasMortgage = encodeYesNo(tokens[13]);
  record.hasDependents = encodeYesNo(tokens[14]);
  record.loanPurpose = encodeLoanPurpose(tokens[15]);
  record.hasCoSigner = encodeYesNo(tokens[16]);

   //  Default (0 = Low Risk, 1 = High Risk)
  record.defaultRisk = encodeYesNo(tokens[17]);
   
  //Validate input Data

  validateInputData(record);

  dataset.push_back(record);
  loadedCount++;

  //Progress indicator for large file;

   if (loadedCount % 50000 == 0) {
            cout << "  Loaded " << loadedCount << " records..." << endl;
        }


}

file.close();

auto endtime = chrono::high_resolution_clock::now();
auto duration = chrono::duration_cast<chrono::seconds>(endtime-startTime);

//Short message for user

    cout << "\nDataset loading completed!" << endl;
    cout << "  Total lines processed: " << lineCount << endl;
    cout << "  Valid records loaded: " << dataset.size() << endl;
    cout << "  Invalid records skipped: " << invalidCount << endl;
    cout << "  Loading time: " << duration.count() << " seconds" << endl;
 

//CAlculate statistics for numerical fellings;

 cout << "\nComputing feature statistics..." << endl;

calculateFeaturesStatistics(dataset);

return dataset;

}

//Data Validation
void validateInputData(LoanRecord& record)
{

    if(record.age < 18) {

        record.age = 18;
    }

    if(record.age > 80){
        record.age = 80;
    }

    if(record.income < 0){
        record.income = 0;
    }

    if(record.income > 5000000){
        record.income = 5000000;
    }

    if(record.creditScore < 300)
    {
        record.creditScore = 300;
    }
    
    if(record.creditScore > 850)
    {
        record.creditScore = 850;
    }

    if(record.dtiRatio < 0.0)
    {
        record.dtiRatio = 0.0;
    }

    if(record.dtiRatio > 1.0)
    {
        record.dtiRatio = 1.0;
    }

    if (record.interestRate < 0.0) record.interestRate = 0.0;
    if (record.interestRate > 30.0) record.interestRate = 30.0;
    
    if (record.monthsEmployed < 0) record.monthsEmployed = 0;
    if (record.monthsEmployed > 600) record.monthsEmployed = 600;
    


}


//Function to calculate data Statistics
void calculateFeaturesStatistics(vector<LoanRecord>&dataset)
{
    int num_features = 16;

    if(dataset.empty()) return;

     vector<vector<double>> featureValues(num_features);

//Not completed yet,, will start work from here

    
    
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
    
    vector<LoanRecord> dataset = loadAndPreprocessDataset(datasetFile);

    if(dataset.empty())
    {
       cerr << "\nError: No data loaded. Please ensure " << datasetFile << " exists." << endl;
        return 1;
    }

}



