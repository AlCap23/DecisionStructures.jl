
"""
    $(TYPEDEF)

    Structure to store decision nodes with additional, computationally expensive objective / reward functions. 
    It decouples the computation of the reward and distributes it evenly over all available processes.

    ## Fields

    $(FIELDS)
"""
struct DecisionTree{T, R, N}
    "The root of the decision tree"
    root::DecisionNode{N}
    "The reward function mapping actions to rewards"
    reward::R
    "The rewards associated with the current action"
    rewards::Vector{T}
end

function DecisionTree(root::DecisionNode{N}, reward::R) where {R, N} 
    T = first(Base.return_types(reward, (Vector{Int},)))
    DecisionTree{T, R, N}(root, reward, T[])
end

Base.in(actions::AbstractVector{Int}, tree::DecisionTree) = Base.in(actions, tree.root)

get_node(tree::DecisionTree, actions::AbstractVector{Int}) = get_node(tree.root, actions)

get_reward(tree::DecisionTree, actions::AbstractVector{Int}) = begin
    node = get_node(tree, actions)
    id = get_reward_index(node)
    iszero(id) ? throw(ErrorException("The node does not have a proper reward.")) : getindex(tree.rewards, id)
end


function add_actions!(f::Function, tree::DecisionTree, actions::AbstractVector{Int})
    (actions ∈ tree.root) && return f() 
    reward_index = size(tree.rewards, 1) + 1
    parent = get_node(tree.root, actions[1:end-1])
    child = DecisionNode(last(actions), DecisionNode[], f(), reward_index)
    push!(tree.rewards, tree.reward(actions))
    push!(children(parent), child)
    return last(tree.rewards)
end

function add_actions!(f::Function, tree::DecisionTree, actions::AbstractVector{Vector{Int}}; pool::AbstractWorkerPool = WorkerPool(1:1))
    reward_idx = zeros(Int, size(actions, 1))
    indicators = zeros(Bool, size(actions, 1))
    offset = size(tree.rewards, 1)
    
    @inbounds for i in axes(actions, 1)
        action = actions[i]

        if action ∈ tree.root
            node_ = get_node(tree, action) 
            reward_idx[i] = get_reward_index(node_)
            continue
        end

        indicators[i] = true
        offset += 1
        reward_idx[i] = offset 
        parent = get_node(tree.root, action[1:end-1])
        child = DecisionNode(last(action), DecisionNode[], f(), reward_idx[i])
        push!(children(parent), child)
    end

    tree.rewards[indicators] = pmap(tree.reward, pool, actions[indicators])

    return tree.rewards[reward_idx]
end




