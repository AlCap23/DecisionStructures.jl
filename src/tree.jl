
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

"""
    $(SIGNATURES)

    Checks if, starting from the root of the provided `tree`, the sequence of `actions` can be found.
"""
Base.in(actions::AbstractVector{Int}, tree::DecisionTree) = Base.in(actions, tree.root)

"""
    $(SIGNATURES)

    Find the descendent of the root of `tree` corresponding to the provided sequence of `actions` starting
    from `node`. If no descendent is found, returns `nothing`.
"""
get_node(tree::DecisionTree, actions::AbstractVector{Int}) = get_node(tree.root, actions)

"""
    $(SIGNATURES)

    Find the reward of the descendent of the root of `tree` corresponding to the provided sequence of `actions` starting
    from `node`. If no descendent is found or no valid reward exists, returns `nothing`.
"""
get_reward(tree::DecisionTree, actions::AbstractVector{Int}) = begin
    node = get_node(tree, actions)
    isnothing(node) && return nothing
    id = get_reward_index(node)
    iszero(id) ? nothing : getindex(tree.rewards, id)
end

"""
    $(SIGNATURES)

    Adds a sequence of `actions` to the `tree`. If no itermediate action is found a new one is created. This new
    action has no reward and uses the `information` returned by `f`.
        
    This is intended to be called using `do` block syntax.
"""
function Base.get!(f::Function, tree::DecisionTree, actions::AbstractVector{Int})
    (actions ∈ tree.root) &&
        return getindex(tree.rewards, get_reward_index(get_node(tree.root, actions)))
    reward_index = size(tree.rewards, 1) + 1

    __get_wo_reward!(f, tree, actions, reward_index)

    push!(tree.rewards, tree.reward(actions))

    return tree.rewards[reward_index]
end

# Simply adds the new sequence to the node w/o computing a reward
function __get_wo_reward!(f::Function,
    tree::DecisionTree{<:Any, <:Any, N},
    actions::AbstractVector{Int},
    reward_index::Int) where {N}
    @assert first(Base.return_types(f))==N "The return type of $f is not compatible with the node information $N"
    current = tree.root
    information = f()
    @inbounds for i in axes(actions, 1)
        action = actions[i]
        id = isempty(children(current)) ? nothing :
             findfirst(==(action), map(get_action, children(current)))
        if isnothing(id)
            child = DecisionNode(action,
                DecisionNode[];
                reward_index = i == size(actions, 1) ? reward_index : 0,
                information...)
            push!(children(current), child)
            current = child
        else
            current = children(current)[id]
        end
    end
    return nothing
end

"""
    $(SIGNATURES)

    Adds a batch of `action`s to the `tree`, behaving like a single `get!` but distributes the computation of the `reward` over the available processes controlled
    by passing in an `AbstractWorkerPool` inside the keyworded arguments (`pool = ...`). Defaults to using the main process.

    This is intended to be called using `do` block syntax.
"""
function Base.get!(f::Function,
    tree::DecisionTree,
    actions::AbstractVector{Vector{Int}};
    pool::AbstractWorkerPool = WorkerPool(1:1))
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
        __get_wo_reward!(f, tree, action, reward_idx[i])
    end

    new_rewards = pmap(tree.reward, pool, actions[indicators])
    append!(tree.rewards, new_rewards)
    return tree.rewards[reward_idx]
end
