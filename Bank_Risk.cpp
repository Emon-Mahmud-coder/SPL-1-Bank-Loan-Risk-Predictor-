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

// ============================================================================
// GLOBAL STATISTICS FOR FEATURE SCALING
// ============================================================================

struct FeatureStats {
    double mean;
    double stddev;
    double min;
    double max;
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

// ============================================================================
// DATA LOADING AND PREPROCESSING
// ============================================================================

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
    cout << "This may take a moment for large datasets..." << endl;
    
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


DecisionTree buildDecisionTree(const vector<LoanRecord>& dataset, int maxDepth, int numFeaturesPerTree, int minSamplesSplit) {
    DecisionTree tree;
    
    // Bootstrap sampling: randomly sample with replacement
    // For very large datasets, use a smaller bootstrap sample
    int bootstrapSize = min((int)dataset.size(), 10000); // Cap at 10K per tree for efficiency
    vector<LoanRecord> bootstrapSample;
    
    for (int i = 0; i < bootstrapSize; i++) {
        int randomIndex = rand() % dataset.size();
        bootstrapSample.push_back(dataset[randomIndex]);
    }
    
    // Random feature selection: choose subset of features
    vector<int> allFeatures;
    for (int i = 0; i < NUM_FEATURES; i++) {
        allFeatures.push_back(i);
    }
    
    // Use modern shuffle instead of deprecated random_shuffle
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

     int maxDepth = 5;
    int minSamplesSplit = 10;
    int numFeatures = NUM_FEATURES;

    DecisionTree tree = buildDecisionTree(
        dataset,
        maxDepth,
        numFeatures,
        minSamplesSplit
    );

    cout << "Decision Tree successfully built!" << endl;
    cout << "Total nodes in tree: " << tree.nodes.size() << endl;
    cout << "Root node index: " << tree.rootIndex << endl;
    
   
    return 0;
}
