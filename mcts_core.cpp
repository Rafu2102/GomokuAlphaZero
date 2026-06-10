#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <unordered_map>

namespace py = pybind11;

struct Node {
    Node* parent;
    std::unordered_map<int, Node*> children;
    int n_visits;
    float q_value;
    float p_prior;
    float original_p;
    
    // MCTS Solver Proofs
    bool is_proven_win;
    bool is_proven_loss;

    Node(Node* p, float prior) : parent(p), n_visits(0), q_value(0.0f), p_prior(prior), original_p(prior), is_proven_win(false), is_proven_loss(false) {}
    ~Node() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }

    bool is_expanded() const {
        return !children.empty();
    }

    float get_value(float c_puct) const {
        if (is_proven_loss) return 10000.0f;
        if (is_proven_win) return -10000.0f;
        
        float u = c_puct * p_prior * std::sqrt((float)(parent ? parent->n_visits : 1)) / (1.0f + n_visits);
        return q_value + u;
    }
};

class CppMCTSTree {
private:
    Node* root;
    float c_puct;
    Node* current_leaf;

public:
    CppMCTSTree(float c_puct) : c_puct(c_puct) {
        root = new Node(nullptr, 1.0f);
        current_leaf = nullptr;
    }
    
    ~CppMCTSTree() {
        delete root;
    }

    std::vector<int> select() {
        Node* node = root;
        std::vector<int> path;
        
        while (node->is_expanded()) {
            float max_val = -1e9f;
            int best_action = -1;
            Node* best_child = nullptr;
            
            for (auto& pair : node->children) {
                float val = pair.second->get_value(c_puct);
                if (val > max_val) {
                    max_val = val;
                    best_action = pair.first;
                    best_child = pair.second;
                }
            }
            if (best_child == nullptr) break;
            
            path.push_back(best_action);
            node = best_child;
        }
        current_leaf = node;
        return path;
    }

    void expand(const std::vector<std::pair<int, float>>& action_probs) {
        if (current_leaf == nullptr) return;
        for (const auto& pair : action_probs) {
            if (current_leaf->children.find(pair.first) == current_leaf->children.end()) {
                current_leaf->children[pair.first] = new Node(current_leaf, pair.second);
            }
        }
    }

    void backup(float leaf_value, bool is_terminal, int winner_perspective) {
        if (current_leaf == nullptr) return;
        
        Node* node = current_leaf;
        float current_val = -leaf_value;
        
        if (is_terminal) {
            if (winner_perspective == 1) { 
                node->is_proven_loss = true;
                node->q_value = -1.0f;
                current_val = 1.0f; 
            } else if (winner_perspective == -1) { 
                node->is_proven_win = true;
                node->q_value = 1.0f;
                current_val = -1.0f;
            } else {
                current_val = 0.0f;
            }
        }

        while (node != nullptr) {
            node->n_visits++;
            
            if (node->is_proven_win) {
                node->q_value = 1.0f;
            } else if (node->is_proven_loss) {
                node->q_value = -1.0f;
            } else {
                node->q_value += (current_val - node->q_value) / node->n_visits;
            }
            
            if (node->is_expanded()) {
                bool all_win = true;
                bool any_loss = false;
                for (auto& pair : node->children) {
                    if (pair.second->is_proven_loss) any_loss = true;
                    if (!pair.second->is_proven_win) all_win = false;
                }
                if (any_loss) {
                    node->is_proven_win = true;
                    node->is_proven_loss = false;
                    node->q_value = 1.0f;
                } else if (all_win) {
                    node->is_proven_loss = true;
                    node->is_proven_win = false;
                    node->q_value = -1.0f;
                }
            }

            node = node->parent;
            current_val = -current_val;
        }
    }

    void update_with_move(int action) {
        if (root->children.find(action) != root->children.end()) {
            Node* new_root = root->children[action];
            root->children.erase(action); 
            delete root;
            root = new_root;
            root->parent = nullptr;
        } else {
            delete root;
            root = new Node(nullptr, 1.0f);
        }
    }

    std::vector<std::pair<int, float>> get_root_action_probs(float temperature) {
        std::vector<std::pair<int, float>> result;
        if (!root->is_expanded()) return result;

        if (temperature < 0.01f) {
            int best_action = -1;
            int max_visits = -1;
            for (auto& pair : root->children) {
                if (pair.second->n_visits > max_visits) {
                    max_visits = pair.second->n_visits;
                    best_action = pair.first;
                }
            }
            if (best_action != -1) {
                result.push_back({best_action, 1.0f});
            }
        } else {
            float sum = 0.0f;
            for (auto& pair : root->children) {
                float v = std::pow((float)pair.second->n_visits, 1.0f / temperature);
                sum += v;
            }
            if (sum > 0) {
                for (auto& pair : root->children) {
                    float v = std::pow((float)pair.second->n_visits, 1.0f / temperature);
                    result.push_back({pair.first, v / sum});
                }
            }
        }
        return result;
    }
    
    void add_dirichlet_noise(float noise_eps, const std::vector<int>& actions, const std::vector<float>& noise) {
        if (actions.size() != noise.size()) return;
        for (size_t i = 0; i < actions.size(); ++i) {
            int action = actions[i];
            auto it = root->children.find(action);
            if (it != root->children.end()) {
                it->second->p_prior = (1.0f - noise_eps) * it->second->original_p + noise_eps * noise[i];
            }
        }
    }

    void apply_vcf_bias(const std::vector<int>& vcf_actions, float vcf_alpha) {
        for (int action : vcf_actions) {
            if (root->children.find(action) != root->children.end()) {
                root->children[action]->p_prior += vcf_alpha;
            }
        }
        
        // Normalize
        float total_prior = 0.0f;
        for (auto& pair : root->children) {
            total_prior += pair.second->p_prior;
        }
        if (total_prior > 0.0f) {
            for (auto& pair : root->children) {
                pair.second->p_prior /= total_prior;
            }
        }
    }

    int get_root_visits() const {
        return root->n_visits;
    }
    
    bool is_root_proven_win() const {
        return root->is_proven_win;
    }
    
    bool is_root_proven_loss() const {
        return root->is_proven_loss;
    }
    
    bool is_child_proven_loss(int action) const {
        auto it = root->children.find(action);
        if (it != root->children.end()) {
            return it->second->is_proven_loss;
        }
        return false;
    }
    
    bool is_root_expanded() const {
        return root->is_expanded();
    }
    
    std::vector<int> get_root_actions() const {
        std::vector<int> acts;
        for (auto& pair : root->children) {
            acts.push_back(pair.first);
        }
        return acts;
    }

    std::vector<int> get_pv_path() const {
        std::vector<int> path;
        Node* node = root;
        for (int i = 0; i < 5; i++) {
            if (!node->is_expanded()) break;
            int best_action = -1;
            int max_v = -1;
            Node* best_child = nullptr;
            for (auto& pair : node->children) {
                if (pair.second->n_visits > max_v) {
                    max_v = pair.second->n_visits;
                    best_action = pair.first;
                    best_child = pair.second;
                }
            }
            if (best_action != -1) {
                path.push_back(best_action);
                node = best_child;
            } else {
                break;
            }
        }
        return path;
    }

    float get_root_q_value() const {
        return root->q_value;
    }

    std::vector<std::pair<int, int>> get_root_visits_list() const {
        std::vector<std::pair<int, int>> result;
        for (auto& pair : root->children) {
            result.push_back({pair.first, pair.second->n_visits});
        }
        return result;
    }
};

PYBIND11_MODULE(mcts_core, m) {
    py::class_<CppMCTSTree>(m, "CppMCTSTree")
        .def(py::init<float>())
        .def("select", &CppMCTSTree::select)
        .def("expand", &CppMCTSTree::expand)
        .def("backup", &CppMCTSTree::backup)
        .def("update_with_move", &CppMCTSTree::update_with_move)
        .def("get_root_action_probs", &CppMCTSTree::get_root_action_probs)
        .def("add_dirichlet_noise", &CppMCTSTree::add_dirichlet_noise)
        .def("apply_vcf_bias", &CppMCTSTree::apply_vcf_bias)
        .def("is_root_proven_win", &CppMCTSTree::is_root_proven_win)
        .def("is_root_proven_loss", &CppMCTSTree::is_root_proven_loss)
        .def("is_child_proven_loss", &CppMCTSTree::is_child_proven_loss)
        .def("is_root_expanded", &CppMCTSTree::is_root_expanded)
        .def("get_root_actions", &CppMCTSTree::get_root_actions)
        .def("get_root_visits", &CppMCTSTree::get_root_visits)
        .def("get_pv_path", &CppMCTSTree::get_pv_path)
        .def("get_root_q_value", &CppMCTSTree::get_root_q_value)
        .def("get_root_visits_list", &CppMCTSTree::get_root_visits_list);
}
