module DecisionStructures

using AbstractTrees
using Distributed
using DocStringExtensions

include("node.jl")
export DecisionNode
export get_node, add_child!, get_action, get_information, get_reward_index
export repeated_get!

include("tree.jl")
export DecisionTree
export get_reward

end
