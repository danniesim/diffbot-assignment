#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <map>
#include <cmath>

struct DataRow {
    std::string person_id, person_name, gender, train_test;
    std::vector<std::string> tokenized_person_name;
};


constexpr auto in_file = "../data/allnames.tsv";
constexpr auto COLUMN_DELIMITER = '\t';
constexpr unsigned int ADDITIVE_SMOOTHING_VALUE = 1;
const std::string TRAIN_LABEL = "Train";
const char DATA_EOL = '\r';

enum DataLabel { male, female };
typedef std::map<std::string, unsigned int> TokenCount;
typedef std::map<std::string, double> TokenValue;

const std::map<std::string, DataLabel> LABEL_TO_ENUM = {
        {"Male", DataLabel::male},
        {"Female", DataLabel::female},
};

const unsigned int NUM_LABELS = LABEL_TO_ENUM.size();

std::string remove_punctuation(const std::string& text) {
    std::string result;
    std::remove_copy_if(text.begin(), text.end(),
                        std::back_inserter(result), //Store output
                        std::ptr_fun<int, int>(&std::ispunct));
    return(result);
}


void count_token(TokenCount& tc, const std::vector<std::string>& tokens) {
    for (auto const& token: tokens) {
        auto it = tc.find(token);
        if (it == tc.end()) {
            tc[token] = ADDITIVE_SMOOTHING_VALUE + 1;
        } else {
            it->second += 1;
        }
    }
}

void print_accuracy(const std::string& header, const unsigned int correct, const unsigned int incorrect) {
    auto total = correct + incorrect;
    std::cout << "## " << header << std::endl;
    std::cout << "correct: " << correct << std::endl;
    std::cout << "incorrect: " << incorrect << std::endl;
    std::cout << "total: " << total << std::endl;
    std::cout << "accuracy: " << (double)correct/ total << std::endl;
}

void predict(const std::vector<DataRow>& data_rows
        , const std::vector<TokenValue>& label_token_values
        , std::vector<unsigned int> label_counts) {
    std::vector<unsigned int> correct(NUM_LABELS);
    std::vector<unsigned int> incorrect(NUM_LABELS);

    for (auto const& row: data_rows) {
        DataLabel truth_label_enum =  LABEL_TO_ENUM.find(row.gender)->second;

        std::vector<double> predicted_values(NUM_LABELS, 0.5);
        for (auto const& token: row.tokenized_person_name) {
            for (auto const& label_enum_map: LABEL_TO_ENUM) {
                auto label_enum = label_enum_map.second;

                double token_value;
                auto it = label_token_values[label_enum].find(token);
                if (it != label_token_values[label_enum].end()) {
                    token_value = it->second;
                } else {
                    token_value = ADDITIVE_SMOOTHING_VALUE / (double)label_counts[label_enum];
                };

                predicted_values[label_enum] = std::log(predicted_values[label_enum]) * std::log(token_value);
            }
        }

        int maxElementIndex = std::max_element(predicted_values.begin(), predicted_values.end()) - predicted_values.begin();

        if (maxElementIndex == truth_label_enum) {
            correct[truth_label_enum]++;
        } else {
            incorrect[truth_label_enum]++;
        }

    }

    unsigned int overall_correct = 0,  overall_incorrect = 0;

    for (auto const& label_enum_map: LABEL_TO_ENUM) {
        auto label_enum = label_enum_map.second;

        print_accuracy(label_enum_map.first, correct[label_enum], incorrect[label_enum]);

        overall_correct += correct[label_enum];
        overall_incorrect += incorrect[label_enum];
    }

    print_accuracy("OVERALL", overall_correct, overall_incorrect);
}


int main() {
    std::cout << "Welcome to Diffbot Gender Classification Assignment" << std::endl;

    // Hold our train and test sets
    std::vector<DataRow> data_train;
    std::vector<DataRow> data_test;

    // File plumbing
    std::ifstream ifs(in_file, std::ifstream::in);
    if (ifs.is_open()) {
        std::cout << "Input file: " << in_file << " opened" << std::endl;
    } else {
        std::cout << "ERROR - Input file: " << in_file << " not found" << std::endl;
        exit(1);
    }

    // Read TSV file into our data sets
    std::string line;
    // Skip 1st line
    std::getline(ifs, line);
    while (!ifs.eof()) {
        std::getline(ifs, line);
        std::stringstream line_ss(line);
        DataRow row;
        std::getline(line_ss, row.person_id, COLUMN_DELIMITER);
        std::getline(line_ss, row.person_name, COLUMN_DELIMITER);
        std::getline(line_ss, row.gender, COLUMN_DELIMITER);
        std::getline(line_ss, row.train_test, DATA_EOL);
//        std::cout << row.person_id << ", " << row.person_name << ", " << row.gender << ", " << row.train_test << std::endl;


        // Sanitize and Tokenize person_name
        std::istringstream iss(remove_punctuation(row.person_name));
        std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                        std::istream_iterator<std::string>{}};

//        for (auto const& token: tokens) {
//            std::cout << token << ", ";
//        }
//        std::cout << std::endl;
        row.tokenized_person_name = tokens;
        if (row.train_test == TRAIN_LABEL) {
            data_train.push_back(row);
        } else {
            data_test.push_back(row);
        }
    }
    ifs.close();

    std::cout << "Num Training Rows: " << data_train.size() << std::endl;
    std::cout << "Num Test Rows: " << data_test.size() << std::endl;

    // Label word counts
    std::vector<TokenCount> label_token_counts(NUM_LABELS);
    std::vector<unsigned int> label_counts(NUM_LABELS, 0);


    for (auto & row: data_test) {
//        std::cout << row.person_id << ", " << row.person_name << ", " << row.gender << ", " << row.train_test << std::endl;

        DataLabel label_enum = LABEL_TO_ENUM.find(row.gender)->second;
        count_token(label_token_counts[label_enum], row.tokenized_person_name);
        label_counts[label_enum]++;
    }

    // Find per label values of each token
    std::vector<TokenValue> label_token_values(NUM_LABELS);

//    std::vector<std::vector<DataLabel>> other_labels_enum(NUM_LABELS);
//    for (auto const& label_enum_map: LABEL_TO_ENUM) {
//        auto label_enum = label_enum_map.second;
//        for (auto const& label_enum_to_add_map: LABEL_TO_ENUM) {
//            auto label_to_add_enum = label_enum_to_add_map.second;
//            if (label_to_add_enum != label_enum) {
//                other_labels_enum[label_enum].push_back(label_to_add_enum);
//            }
//        }
//    }

    for (auto const& label_enum_map: LABEL_TO_ENUM) {
        auto label_enum = label_enum_map.second;

        for (auto const& token_count: label_token_counts[label_enum]) {
            label_token_values[label_enum][token_count.first] = \
                (double)token_count.second / (double)label_counts[label_enum];

//            for (auto const& other_label_enum: other_labels_enum[label_enum]) {
//                auto it = label_token_values[other_label_enum].find(token_count.first);
//            }
        }
    }

    std::cout << std::endl << "# TRAIN DATA SET" << std::endl ;
    predict(data_train, label_token_values, label_counts);

    std::cout << std::endl << "# TEST DATA SET" << std::endl ;
    predict(data_test, label_token_values, label_counts);

    return 0;
}
