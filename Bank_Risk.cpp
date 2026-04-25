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
    int education;           // encoded: 0=High School, 1=Bachelor, 2=Master, 3=PhD
    int employmentType;      // encoded: 0=Unemployed, 1=Self-employed, 2=Full-time, 3=Part-time
    int maritalStatus;       // encoded: 0=Single, 1=Married, 2=Divorced
    int hasMortgage;         // 0 or 1
    int hasDependents;       // 0 or 1
    int loanPurpose;         // encoded: 0=Auto, 1=Business, 2=Education, 3=Home, 4=Other
    int hasCoSigner;         // 0 or 1
    int defaultRisk;         // Target: 0=Low Risk, 1=High Risk
};

 int NUM_FEATURES = 16;

struct DecisionNode {
    bool isLeaf;
    int predictedClass;      // For leaf nodes: 0=Low Risk, 1=High Risk
    int featureIndex;        // Which feature to split on
    double threshold;        // Split threshold
    int leftChild;           // Index of left child in tree nodes vector
    int rightChild;          // Index of right child in tree nodes vector
};

struct DecisionTree {
    vector<DecisionNode> nodes;
    int rootIndex;
};

struct RandomForest {
    vector<DecisionTree> trees;
    int numTrees;
};


// Global statistics for feature importance


struct FeatureStats {
    double mean;
    double stddev;
    double min;
    double max;
};

struct RiskAnalysis{
int predictedClass;
    double confidence;
    vector<string> riskFactors;
    vector<string> positiveFactors;
    string riskLevel;
    vector<string> recommendations;
};

struct ModelMetrics {
    double accuracy;
    double precision;
    double recall;
    double f1Score;
    int truePositive, trueNegative, falsePositive, falseNegative;
};

map<int, FeatureStats> featureStats;
map<int, int> featureUsageCount;  // Track how often each feature is used in splits

// Feature names for reporting
const vector<string> FEATURE_NAMES = {
    "Age", "Income", "Loan Amount", "Credit Score", 
    "Months Employed", "Credit Lines", "Interest Rate", "Loan Term",
    "DTI Ratio", "Education", "Employment Type", "Marital Status",
    "Has Mortgage", "Has Dependents", "Loan Purpose", "Has Co-Signer"
};






// Get current timestamp for logging
string getCurrentTimestamp() {
    time_t now = time(0);
    char buf[80];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return string(buf);
}

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

// Decode employment type
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

// Decode marital status
string getMaritalStatusName(int code) {
    switch(code) {
        case 0: return "Single";
        case 1: return "Married";
        case 2: return "Divorced";
        default: return "Unknown";
    }
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

// Validate credit score
bool isValidCreditScore(int score) {
    return score >= 300 && score <= 850;
}

// Validate DTI ratio
bool isValidDTIRatio(double ratio) {
    return ratio >= 0.0 && ratio <= 1.0;
}

// Validate age
bool isValidAge(int age) {
    return age >= 18 && age <= 80;
}

// Validate income
bool isValidIncome(double income) {
    return income >= 0 && income <= 500000;
}


void validateInputData(LoanRecord& record) {
   
    if (record.age < 18) record.age = 18;
    if (record.age > 80) record.age = 80;
    
    if (record.income < 0) record.income = 0;
    if (record.income > 500000) record.income = 500000;
    
    if (record.creditScore < 300) record.creditScore = 300;
    if (record.creditScore > 850) record.creditScore = 850;
    
    if (record.dtiRatio < 0.0) record.dtiRatio = 0.0;
    if (record.dtiRatio > 1.0) record.dtiRatio = 1.0;
    
    if (record.interestRate < 0.0) record.interestRate = 0.0;
    if (record.interestRate > 30.0) record.interestRate = 30.0;
    
    if (record.monthsEmployed < 0) record.monthsEmployed = 0;
    if (record.monthsEmployed > 600) record.monthsEmployed = 600;
}


// Data loading and preprocessing


vector<LoanRecord> loadAndPreprocessDataset(const string& filename) {
    vector<LoanRecord> dataset;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return dataset;
    }
    
    string line;
    bool isFirstLine = true;
    int lineCount = 0;
    int loadedCount = 0;
    int invalidCount = 0;
    
    cout << "Loading dataset from " << filename << "..." << endl;
    cout << "This may take a moment..." << endl;
    
    auto startTime = chrono::high_resolution_clock::now();
    
    while (getline(file, line)) {
        lineCount++;
        
        // Skip header row
        if (isFirstLine) {
            isFirstLine = false;
            continue;
        }
        
        // Skip empty lines
        if (line.empty()) continue;
        
        vector<string> tokens = splitCSV(line);
        
        // Ensure we have enough columns (at least 18 columns including LoanID)
        if (tokens.size() < 18) {
            invalidCount++;
            continue;
        }
        
        LoanRecord record;
        
        // Parse and convert each field
        // Skip LoanID (tokens[0])
        record.age = safeStoi(tokens[1], 30);
        record.income = safeStod(tokens[2], 50000.0);
        record.loanAmount = safeStoi(tokens[3], 10000);
        record.creditScore = safeStoi(tokens[4], 650);
        record.monthsEmployed = safeStoi(tokens[5], 12);
        record.numCreditLines = safeStoi(tokens[6], 2);
        record.interestRate = safeStod(tokens[7], 5.0);
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
        
        // Validate data
        validateInputData(record);
        
        dataset.push_back(record);
        loadedCount++;
        
        // Progress indicator for large files
        if (loadedCount % 50000 == 0) {
            cout << "  Loaded " << loadedCount << " records..." << endl;
        }
    }
    
    file.close();
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(endTime - startTime);
    
    cout << "Dataset loading completed!" << endl;
    cout << "Total lines processed: " << lineCount << endl;
    cout << "Valid records loaded: " << dataset.size() << endl;
    cout << "Invalid records skipped: " << invalidCount << endl;
    cout << "Loading time: " << duration.count() << " seconds" << endl;
    
    
    
    
    return dataset;
}

double getFeatureValue(LoanRecord& record , int featureIndex)
{
    switch(featureIndex){

        case 0: return record.age;
        case 1: return record.income;
        case 2: return record.loanAmount;
        case 3: return record.creditScore;
        case 4: return record.monthsEmployed;
        case 5: return record.numCreditLines;
        case 6: return record.interestRate;
        case 7: return record.loanTerm;
        case 8: return record.dtiRatio;
        case 9: return record.education;
        case 10: return record.employmentType;
        case 11: return record.maritalStatus;
        case 12: return record.hasMortgage;
        case 13: return record.hasDependents;
        case 14: return record.loanPurpose;
        case 15: return record.hasCoSigner;

        default: return 0.0;

    }
}

//Function to calculate gini impurity
double calculateGini(vector<LoanRecord>& records){

    if (records.empty()) return 0.0;
    
    int countLowRisk = 0;
    int countHighRisk = 0;

    for(auto& record:records){

        if(record.defaultRisk == 0) countLowRisk++;
        else countHighRisk++;
    }

    double probLow = (double)countLowRisk / records.size();
    double probHigh = (double)countHighRisk / records.size();
    
    return 1.0 - (probLow*probLow + probHigh*probHigh);

}

// get majority class in a set of records

int getMajorityClass(vector<LoanRecord>& records){

    if(records.empty()) return 0.0;

    int countLowRisk = 0;
    int countHighRisk = 0;
    
    for (const auto& record : records) {
        if (record.defaultRisk == 0) countLowRisk++;
        else countHighRisk++;
    }
    
    return (countHighRisk > countLowRisk) ? 1 : 0;
}

struct SplitInfo
{
  int featureIndex;
    double threshold;
    double giniGain;
    vector<LoanRecord> leftSplit;
    vector<LoanRecord> rightSplit;   /* data */
};

SplitInfo findBestSplit(vector<LoanRecord>& records, vector<int>& availableFeatures) {
    SplitInfo bestSplit;
    bestSplit.giniGain = -1.0;
    bestSplit.featureIndex = -1;
    
    double parentGini = calculateGini(records);
    
    // Try each available feature
    for (int featureIndex : availableFeatures) {
        // Collect all unique values for this feature
        vector<double> values;
        for ( auto& record : records) {
            values.push_back(getFeatureValue(record, featureIndex));
        }
        
        sort(values.begin(), values.end());
        values.erase(unique(values.begin(), values.end()), values.end());
        
        // For large datasets, sample thresholds to speed up training
        int maxSamples = 20; // Try up to 20 thresholds per feature
        int step = max(1, (int)values.size() / maxSamples);
        
        // Try splits at sampled points
        for (size_t i = 0; i < values.size() - 1; i += step) {
            double threshold = (values[i] + values[min(i + 1, values.size() - 1)]) / 2.0;
            
            vector<LoanRecord> leftSplit, rightSplit;
            
            for ( auto& record : records) {
                if (getFeatureValue(record, featureIndex) <= threshold) {
                    leftSplit.push_back(record);
                } else {
                    rightSplit.push_back(record);
                }
            }
            
            if (leftSplit.empty() || rightSplit.empty()) continue;
            
            // Calculate weighted Gini impurity
            double leftGini = calculateGini(leftSplit);
            double rightGini = calculateGini(rightSplit);
            double weightedGini = (leftSplit.size() * leftGini + rightSplit.size() * rightGini) / records.size();
            
            double giniGain = parentGini - weightedGini;
            
            if (giniGain > bestSplit.giniGain) {
                bestSplit.featureIndex = featureIndex;
                bestSplit.threshold = threshold;
                bestSplit.giniGain = giniGain;
                bestSplit.leftSplit = leftSplit;
                bestSplit.rightSplit = rightSplit;
            }
        }
    }
    
    // Track feature usage for importance calculation
    if (bestSplit.featureIndex >= 0) {
        featureUsageCount[bestSplit.featureIndex]++;
    }
    
    return bestSplit;
}

// Recursively build decision tree
int buildTreeRecursive(DecisionTree& tree, vector<LoanRecord>& records, 
                       vector<int>& availableFeatures, int depth, int maxDepth, int minSamplesSplit) {
    
    DecisionNode node;
    int currentIndex = tree.nodes.size();
    tree.nodes.push_back(node); // Reserve space
    
    // Stopping criteria
    if (records.empty() || depth >= maxDepth || records.size() < minSamplesSplit || 
        calculateGini(records) < 0.01 || availableFeatures.empty()) {
        tree.nodes[currentIndex].isLeaf = true;
        tree.nodes[currentIndex].predictedClass = getMajorityClass(records);
        return currentIndex;
    }
    
    // Find best split
    SplitInfo split = findBestSplit(records, availableFeatures);
    
    // If no good split found, make leaf node
    if (split.giniGain <= 0.001) {
        tree.nodes[currentIndex].isLeaf = true;
        tree.nodes[currentIndex].predictedClass = getMajorityClass(records);
        return currentIndex;
    }
    
    // Create internal node
    tree.nodes[currentIndex].isLeaf = false;
    tree.nodes[currentIndex].featureIndex = split.featureIndex;
    tree.nodes[currentIndex].threshold = split.threshold;
    
    // Recursively build left and right subtrees
    int leftChildIndex = buildTreeRecursive(tree, split.leftSplit, availableFeatures, depth + 1, maxDepth, minSamplesSplit);
    int rightChildIndex = buildTreeRecursive(tree, split.rightSplit, availableFeatures, depth + 1, maxDepth, minSamplesSplit);
    
    tree.nodes[currentIndex].leftChild = leftChildIndex;
    tree.nodes[currentIndex].rightChild = rightChildIndex;
    
    return currentIndex;
}


DecisionTree buildDecisionTree(vector<LoanRecord>& dataset, int maxDepth, int numFeaturesPerTree, int minSamplesSplit) {
    DecisionTree tree;

 
    // Separate records by class
    vector<LoanRecord> lowRiskRecords, highRiskRecords;
    for (auto& r : dataset) {
        if (r.defaultRisk == 0) lowRiskRecords.push_back(r);
        else highRiskRecords.push_back(r);
    }

    // Sample equally from both classes (50/50 balance per tree)
    int halfSize = min(5000, (int)min(lowRiskRecords.size(), highRiskRecords.size()));
    vector<LoanRecord> bootstrapSample;

    for (int i = 0; i < halfSize; i++) {
        bootstrapSample.push_back(lowRiskRecords[rand() % lowRiskRecords.size()]);
        bootstrapSample.push_back(highRiskRecords[rand() % highRiskRecords.size()]);
    }

    // Random feature selection: choose subset of features
    vector<int> allFeatures;
    for (int i = 0; i < NUM_FEATURES; i++) {
        allFeatures.push_back(i);
    }

    // Shuffle features
    random_device rd;
    mt19937 g(rd());
    shuffle(allFeatures.begin(), allFeatures.end(), g);

    vector<int> selectedFeatures;
    for (int i = 0; i < numFeaturesPerTree && i < NUM_FEATURES; i++) {
        selectedFeatures.push_back(allFeatures[i]);
    }

    // Build the tree
    tree.rootIndex = buildTreeRecursive(tree, bootstrapSample, selectedFeatures, 0, maxDepth, minSamplesSplit);

    return tree;
}
RandomForest buildRandomForest( vector<LoanRecord>& dataset,int numTrees,int maxDepth,int numFeaturesPerTree,int minSamplesSplit) {
    RandomForest forest;
    forest.numTrees = numTrees;

    int progressStep = max(1, numTrees / 10);  // 10%, 20%, ...
     
    cout << "\nRandom Forest building...."<<endl;
    for (int i = 0; i < numTrees; i++) {
        DecisionTree tree = buildDecisionTree(
            dataset,
            maxDepth,
            numFeaturesPerTree,
            minSamplesSplit
        );

        forest.trees.push_back(tree);

        // output
        if ((i + 1) % progressStep == 0 || i + 1 == numTrees) {
            cout << (i + 1) << "/" << numTrees << " trees built" << endl;
        }
    }

    return forest;
}

RandomForest trainRandomForest(vector<LoanRecord>& dataset , int numTrees , int maxDepth , int minSamplesSplit){

    cout << "\n=== Training Random Forest ====" << endl;
    cout << "Number of Tress: " <<numTrees << endl;
    cout << "Max Depth Per Tree: " << maxDepth << endl;
    cout << "Min samples to split: " << minSamplesSplit << endl;
    cout << "Training dataset size: " << dataset.size() << endl;

    int numFeaturesPerTree = sqrt(NUM_FEATURES);
    cout << "Features per tree split: " << numFeaturesPerTree << endl;

    // Class Distribution

    int low = 0 , high = 0;

    for(auto& r:dataset){
        if(r.defaultRisk == 0) low++;
        else high++;
    }
    
    cout << "Class Distribution:" << endl;
    cout << "Low risk: " << low << endl;
    cout << "High risk: " << high << endl;

    //Start time counting

    auto start = chrono::high_resolution_clock::now();

    RandomForest forest = buildRandomForest(
        dataset ,
         numTrees , 
         maxDepth ,
          numFeaturesPerTree , 
          minSamplesSplit
        );

        auto end = chrono:: high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end - start);
        
         cout << "Random Forest training completed!" << endl;
         cout << "Training time: " << duration.count() << " seconds" << endl;

         return forest;

}





void displayRiskAnalysis(RiskAnalysis& analysis){
 
    cout << "\n=== Detailed Risk Analysis ===" << endl;
    cout << "Risk Level: " << analysis.riskLevel << endl;
    cout << "Confidence: " << fixed << setprecision(1) << analysis.confidence << "%" << endl;
    
    if (!analysis.riskFactors.empty()) {
        cout << "\nRisk Factors Identified:" << endl;
        for (const auto& factor : analysis.riskFactors) {
            cout << "  ⚠ " << factor << endl;
        }
    }

   if (!analysis.recommendations.empty()) {
        cout << "\n" << string(60, '=') << endl;
        for (const auto& rec : analysis.recommendations) {
            cout << rec << endl;
        }
    } 




}


LoanRecord getUserInput(){

    LoanRecord record;
    cout << "\n=== Enter Loan Applicant Information ===" << endl;
    
    cout << "Age: ";
    cin >> record.age;
    
    cout << "Annual Income: ";
    cin >> record.income;

    cout << "Loan Amount: " ;
    cin >> record.loanAmount; 

    cout << "Credit Score: ";
    cin >> record.creditScore;
    
    cout << "Months Employed: ";
    cin >> record.monthsEmployed;

     cout << "Number of Credit Lines: ";
    cin >> record.numCreditLines;
    
    cout << "Interest Rate (%): ";
    cin >> record.interestRate;

    cout << "Loan Term (months): ";
    cin >> record.loanTerm;
    
    cout << "Debt-to-Income Ratio: ";
    cin >> record.dtiRatio;

    cout << "\nEducation (0=High School, 1=Bachelor's, 2=Master's, 3=PhD): ";
    cin >> record.education;
    
    cout << "Employment Type (0=Unemployed, 1=Self-employed, 2=Full-time, 3=Part-time): ";
    cin >> record.employmentType;
    
    cout << "Marital Status (0=Single, 1=Married, 2=Divorced): ";
    cin >> record.maritalStatus;
    
    cout << "Has Mortgage? (0=No, 1=Yes): ";
    cin >> record.hasMortgage;

     cout << "Has Dependents? (0=No, 1=Yes): ";
    cin >> record.hasDependents;
    
    cout << "Loan Purpose (0=Auto, 1=Business, 2=Education, 3=Home, 4=Other): ";
    cin >> record.loanPurpose;

    cout << "Has Co-Signer? (0=No, 1=Yes): ";
    cin >> record.hasCoSigner;
    
    // Validate input
    validateInputData(record);
    
    // We don't ask for the default risk (target) - we're predicting it!
    record.defaultRisk = -1; // Unknown
    
    return record;

}


void displayApplicantSummary(const LoanRecord& record){

cout << "\n===Applicant Profile summary ===" << endl;
cout <<"Personal Information:" << endl;

 cout << "  Age: " << record.age << " years" << endl;
 cout << "  Education: " << getEducationName(record.education) << endl;
 cout << "  Marital Status: " << getMaritalStatusName(record.maritalStatus) << endl;
 cout << "  Has Dependents: " << (record.hasDependents ? "Yes" : "No") << endl;
    
 cout << "\n Financial Infromation:" << endl;

 cout << "  Annual Income: $" << fixed << setprecision(0) << record.income << endl;
 cout << "  Credit Score: " << record.creditScore << endl;
 cout << "  DTI Ratio: " << fixed << setprecision(2) << (record.dtiRatio * 100) << "%" << endl;
 cout << "  Credit Lines: " << record.numCreditLines << endl;

  cout << "\nEmployment:" << endl;
  cout << "  Type: " << getEmploymentName(record.employmentType) << endl;
  cout << "  Duration: " << record.monthsEmployed << " months" << endl;
  
  cout << "\nLoan Details:" << endl;
  cout << "  Amount: $" << fixed << setprecision(0) << record.loanAmount << endl;
  cout << "  Purpose: " << getLoanPurposeName(record.loanPurpose) << endl;
  cout << "  Term: " << record.loanTerm << " months" << endl;
  cout << "  Interest Rate: " << fixed << setprecision(2) << record.interestRate << "%" << endl;
  cout << "  Has Co-Signer: " << (record.hasCoSigner ? "Yes" : "No") << endl;
  cout << "  Has Mortgage: " << (record.hasMortgage ? "Yes" : "No") << endl;

}

//Prediction
// Predict class for a single record using one decision tree

int predictWithTree(const DecisionTree& tree ,  LoanRecord& record){

 int currentIndex = tree.rootIndex;

 while (true)
 {
    const DecisionNode& node = tree.nodes[currentIndex];

    if(node.isLeaf)
    {
        return node.predictedClass;
    }

    double featureValue = getFeatureValue(record , node.featureIndex);
    
    if(featureValue <= node.threshold){
        currentIndex = node.leftChild;
    }else{
        currentIndex = node.rightChild;
    }

 }
 return 0; // default, should never reach here
 
}

// Predict class using Random Forest (majority voting)
struct PredictionResult {
    int predictedClass;
    int votesLowRisk;
    int votesHighRisk;
    double confidence;
};

PredictionResult predictWithRandomForest(const RandomForest& forest , LoanRecord& record){
PredictionResult result;

result.votesLowRisk = 0;
result.votesHighRisk = 0;

for(const auto & tree:forest.trees){

    int prediction = predictWithTree(tree , record);
    if(prediction == 0)result.votesLowRisk++;
    else result.votesHighRisk++;

}

result.predictedClass = (result.votesHighRisk > result.votesLowRisk)? 1:0;
result.confidence = (double)max(result.votesLowRisk, result.votesHighRisk) / forest.numTrees * 100.0;
    
return result;

}

ModelMetrics evaluateModel(const RandomForest& forest, vector<LoanRecord>& testSet) {
    ModelMetrics m = {0};
    for (auto& record : testSet) {
        PredictionResult pred = predictWithRandomForest(forest, const_cast<LoanRecord&>(record));
        int actual = record.defaultRisk;
        int predicted = pred.predictedClass;
        if (actual == 1 && predicted == 1) m.truePositive++;
        else if (actual == 0 && predicted == 0) m.trueNegative++;
        else if (actual == 0 && predicted == 1) m.falsePositive++;
        else if (actual == 1 && predicted == 0) m.falseNegative++;
    }
    m.accuracy  = 100.0 * (m.truePositive + m.trueNegative) / testSet.size();
    m.precision = (m.truePositive + m.falsePositive) > 0 ?
                  100.0 * m.truePositive / (m.truePositive + m.falsePositive) : 0;
    m.recall    = (m.truePositive + m.falseNegative) > 0 ?
                  100.0 * m.truePositive / (m.truePositive + m.falseNegative) : 0;
    m.f1Score   = (m.precision + m.recall) > 0 ?
                  2 * m.precision * m.recall / (m.precision + m.recall) : 0;
    return m;
}

void displayMetrics(const ModelMetrics& m) {
    cout << "\n========= MODEL EVALUATION =========" << endl;
    cout << "Confusion Matrix:" << endl;
    cout << "                  Predicted" << endl;
    cout << "                  Low    High" << endl;
    cout << "Actual  Low    [  " << m.trueNegative << "    " << m.falsePositive << "  ]" << endl;
    cout << "        High   [  " << m.falseNegative << "    " << m.truePositive << "  ]" << endl;
    cout << "\nAccuracy:  " << fixed << setprecision(2) << m.accuracy << "%" << endl;
    cout << "Precision: " << m.precision << "%" << endl;
    cout << "Recall:    " << m.recall << "%" << endl;
    cout << "F1 Score:  " << m.f1Score << "%" << endl;
    cout << "=====================================" << endl;
}

vector<string>generateRecommendations(LoanRecord& record , int prediciton){

vector<string> recommendations;
if(prediciton == 1){
    recommendations.push_back("RECOMMENDATION: Loan application should be carefully reviewed");

        if (record.creditScore < 650) {
            recommendations.push_back("- Consider requiring a higher down payment");
            recommendations.push_back("- Recommend credit counseling before approval");
        }
          if (record.dtiRatio > 0.45) {
            recommendations.push_back("- Debt consolidation should be considered");
            recommendations.push_back("- Verify all income sources thoroughly");
        }
        if (record.hasCoSigner == 0) {
            recommendations.push_back("- Strongly recommend obtaining a co-signer");
        }
         if (record.employmentType != 2) {
            recommendations.push_back("- Request additional proof of income stability");
        }

        recommendations.push_back("- Consider increasing interest rate to offset risk");
        recommendations.push_back("- Implement closer monitoring during loan term");


}else { // Low Risk
        recommendations.push_back("RECOMMENDATION: Loan application shows strong indicators");
        
        if (record.creditScore >= 750 && record.dtiRatio < 0.35) {
            recommendations.push_back("- Eligible for preferred interest rates");
            recommendations.push_back("- Fast-track approval recommended");
        }
        
        if (record.hasCoSigner == 1) {
            recommendations.push_back("- Co-signer may not be necessary given strong profile");
        }
        
        recommendations.push_back("- Standard monitoring procedures sufficient");
        recommendations.push_back("- Consider for future credit limit increases");
    }

    return recommendations;

}

//Risk Analysis

RiskAnalysis analyzeRisk(LoanRecord& record , int prediction , double confidence){

RiskAnalysis analysis;
analysis.riskLevel = prediction == 1 ? "VERY HIGH RISK" : "VERY LOW RISK";
analysis.confidence = confidence;

//Determine Risk level based on confidence

if(confidence >= 90){
     analysis.riskLevel = prediction == 1 ? "VERY HIGH RISK" : "VERY LOW RISK";
}else if(confidence >=75){
      analysis.riskLevel = prediction == 1 ? "HIGH RISK" : "LOW RISK";
}else if(confidence >=60){
    analysis.riskLevel = prediction == 1 ? "MODERATE-HIGH RISK" : "MODERATE-LOW RISK";

} else {
        analysis.riskLevel = "UNCERTAIN - MANUAL REVIEW NEEDED";
    }
//Identify risk factors

if(record.creditScore < 600)
{
    analysis.riskFactors.push_back("Poor credit score (" + to_string(record.creditScore) + ")");
}

 if (record.dtiRatio > 0.50) {
        analysis.riskFactors.push_back("High debt-to-income ratio (" + to_string((int)(record.dtiRatio * 100)) + "%)");
 }
 if (record.employmentType == 0) {
        analysis.riskFactors.push_back("Currently unemployed");
 }

 if (record.interestRate > 10.0) {
        analysis.riskFactors.push_back("High interest rate (" + to_string(record.interestRate) + "%)");
 }
 if (record.monthsEmployed < 12) {
        analysis.riskFactors.push_back("Short employment history (" + to_string(record.monthsEmployed) + " months)");
    }
    if (record.hasCoSigner == 0 && record.creditScore < 650) {
        analysis.riskFactors.push_back("No co-signer with low credit");
    }

// Identify positive factors
if (record.creditScore >= 750) {
        analysis.positiveFactors.push_back("Excellent credit score (" + to_string(record.creditScore) + ")");
}
if (record.dtiRatio < 0.35) {
        analysis.positiveFactors.push_back("Low debt-to-income ratio (" + to_string((int)(record.dtiRatio * 100)) + "%)");
}
if (record.employmentType == 2 && record.monthsEmployed >= 36) {
        analysis.positiveFactors.push_back("Stable full-time employment");
}
if (record.hasCoSigner == 1) {
        analysis.positiveFactors.push_back("Has co-signer");
    }
if (record.education >= 2) {
        analysis.positiveFactors.push_back("Advanced degree (" + getEducationName(record.education) + ")");
}

if (record.income >= 80000) {
        analysis.positiveFactors.push_back("High income ($" + to_string((int)record.income) + ")");
}

//Generate Recommendetaion
analysis.recommendations = generateRecommendations(record , prediction);


return analysis;

}

void displayDetailedStatistics(const vector<LoanRecord>& dataset) {
    cout << "\n=== Detailed Dataset Statistics ===" << endl;
    
    // Class distribution
    int countLow = 0, countHigh = 0;
    for (const auto& record : dataset) {
        if (record.defaultRisk == 0) countLow++;
        else countHigh++;
    }
    
    cout << "\nClass Distribution:" << endl;
    cout << "  Low Risk:  " << countLow << " (" 
         << fixed << setprecision(1) << (100.0 * countLow / dataset.size()) << "%)" << endl;
    cout << "  High Risk: " << countHigh << " (" 
         << (100.0 * countHigh / dataset.size()) << "%)" << endl;
    
    // Feature statistics by class
    double avgCreditLow = 0, avgCreditHigh = 0;
    double avgDtiLow = 0, avgDtiHigh = 0;
    double avgIncomeLow = 0, avgIncomeHigh = 0;
    double avgInterestLow = 0, avgInterestHigh = 0;
    
    for (const auto& record : dataset) {
        if (record.defaultRisk == 0) {
            avgCreditLow += record.creditScore;
            avgDtiLow += record.dtiRatio;
            avgIncomeLow += record.income;
            avgInterestLow += record.interestRate;
        } else {
            avgCreditHigh += record.creditScore;
            avgDtiHigh += record.dtiRatio;
            avgIncomeHigh += record.income;
            avgInterestHigh += record.interestRate;
        }
    }
    
    if (countLow > 0) {
        avgCreditLow /= countLow;
        avgDtiLow /= countLow;
        avgIncomeLow /= countLow;
        avgInterestLow /= countLow;
    }
    
    if (countHigh > 0) {
        avgCreditHigh /= countHigh;
        avgDtiHigh /= countHigh;
        avgIncomeHigh /= countHigh;
        avgInterestHigh /= countHigh;
    }
    
    cout << "\nKey Metrics by Risk Level:" << endl;
    cout << "  Credit Score    - Low Risk: " << fixed << setprecision(0) << avgCreditLow 
         << ", High Risk: " << avgCreditHigh << endl;
    cout << "  DTI Ratio       - Low Risk: " << fixed << setprecision(3) << avgDtiLow 
         << ", High Risk: " << avgDtiHigh << endl;
    cout << "  Income          - Low Risk: $" << fixed << setprecision(0) << avgIncomeLow 
         << ", High Risk: $" << avgIncomeHigh << endl;
    cout << "  Interest Rate   - Low Risk: " << fixed << setprecision(2) << avgInterestLow 
         << "%, High Risk: " << avgInterestHigh << "%" << endl;
    
    // Employment distribution
    map<int, int> employmentDist;
    for (const auto& record : dataset) {
        employmentDist[record.employmentType]++;
    }
    
    cout << "\nEmployment Type Distribution:" << endl;
    for (const auto& pair : employmentDist) {
        cout << "  " << getEmploymentName(pair.first) << ": " 
             << pair.second << " (" << fixed << setprecision(1) 
             << (100.0 * pair.second / dataset.size()) << "%)" << endl;
    }
    
    // Loan purpose distribution
    map<int, int> purposeDist;
    for (const auto& record : dataset) {
        purposeDist[record.loanPurpose]++;
    }
    
    cout << "\nLoan Purpose Distribution:" << endl;
    for (const auto& pair : purposeDist) {
        cout << "  " << getLoanPurposeName(pair.first) << ": " 
             << pair.second << " (" << fixed << setprecision(1) 
             << (100.0 * pair.second / dataset.size()) << "%)" << endl;
    }
}



//Main Program

int main() {
    
    // Initialize random seed
    srand(time(0));
    
    cout << "=========================================" << endl;
    cout << "  LOAN RISK PREDICTION SYSTEM" << endl;
    cout << "  Using Random Forest Algorithm" << endl;
    cout << "=========================================" << endl;
    
    // Step 1: Load and preprocess dataset
    string datasetFile = "data.csv";
    vector<LoanRecord> dataset = loadAndPreprocessDataset(datasetFile);
    
    if (dataset.empty()) {
        cerr << "\nError: No data loaded. Please ensure " << datasetFile << " exists." << endl;
        return 1;
    }
     //Step 1.a: Display detailed statistics about the dataset
    displayDetailedStatistics(dataset);

    random_shuffle(dataset.begin(), dataset.end());
    int trainSize = dataset.size() * 0.8;
    vector<LoanRecord> trainSet(dataset.begin(), dataset.begin() + trainSize);
    vector<LoanRecord> testSet(dataset.begin() + trainSize, dataset.end());

int numTrees = 100;
int maxDepth = 15;
int minSamplesSplit = 10;
int numFeaturesPerTree = sqrt(NUM_FEATURES);

RandomForest forest = trainRandomForest(
    trainSet,
    numTrees,
    maxDepth,
    minSamplesSplit
);

cout << "\n=========================================" << endl;
cout << " Random Forest training Completed" << endl;
cout << " Total trees built: " << forest.numTrees << endl;
cout << "=========================================" << endl;

ModelMetrics metrics = evaluateModel(forest, testSet);
displayMetrics(metrics);


//Get user input and make prediction

char continueInput = 'y';

while (continueInput == 'y' || continueInput == 'Y')
{
    LoanRecord applicant = getUserInput();
    displayApplicantSummary(applicant);
    cout << "\n===Making Prediction===" <<endl;
    cout << "consulting " << forest.numTrees << " decision trees...." << endl;
    
    PredictionResult prediction = predictWithRandomForest(forest , applicant);

     cout << "\n=========================================" << endl;
     cout << "Prediction Results:" << endl;
     cout << "=========================================" << endl;

     cout << "Voting Results:" << endl;
     cout << "  Low Risk votes:  " << prediction.votesLowRisk << " / " << forest.numTrees << endl;
     cout << "  High Risk votes: " << prediction.votesHighRisk << " / " << forest.numTrees << endl;
     cout << "  Confidence: " << fixed << setprecision(1) << prediction.confidence << "%" << endl;
     cout << endl;

     if(prediction.predictedClass == 0){
         cout << "✓ LOW RISK - Loan application is likely to be APPROVED" << endl;

         if(prediction.confidence < 60){
              cout << "  Note: Moderate confidence. Review carefully." << endl;
         }else if(prediction.confidence > 80){
             cout << "  Strong confidence in prediction." << endl;
         }    

     }else{
         cout << "✗ HIGH RISK - Loan application may be REJECTED" << endl;
         if (prediction.confidence < 60) {
                cout << "  Note: Moderate confidence. Additional review recommended." << endl;
            }else if(prediction.confidence >=80){
                 cout << "  Strong confidence in prediction." << endl;
            }
     }

     cout << "=========================================" << endl;

     //Perform detailed risk analysis
     RiskAnalysis riskAnalysis = analyzeRisk(applicant , prediction.predictedClass , prediction.confidence);
     displayRiskAnalysis(riskAnalysis);

     cout << "\nDo you want to predict for another applicant? (y/n): ";
     cin >> continueInput;
     
     
}

    cout << "\n=========================================" << endl;
    cout << "Thank you for using the Loan Risk Prediction System!" << endl;
    cout << "Total predictions made: " << (continueInput == 'n' || continueInput == 'N' ? "Session complete" : "") << endl;
    cout << "=========================================" << endl;

   
    return 0;
}
