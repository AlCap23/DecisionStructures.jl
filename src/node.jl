using AbstractTrees

struct DecisionNode{N <: NamedTuple} <: AbstractNode{N}
    "The current action"
    action::Int
    "The children of the node"
    children::Vector{DecisionNode}
    "All additional information needed"
    information::N
    "An additional field used for expensive computations"
    reward_index::Int
end

function DecisionNode(action::Int, children::Vector{DecisionNode} = DecisionNode[]; reward_index::Int = 0, kwargs...)
    information = NamedTuple(kwargs)
    DecisionNode{typeof(information)}(action, children, information, reward_index)
end

AbstractTrees.children(node::DecisionNode) = getfield(node, :children)
AbstractTrees.nodevalue(node::DecisionNode) = getfield(node, :information)

get_action(node::DecisionNode) = getfield(node, :action)
get_information(node::DecisionNode) = getfield(node, :information)
get_reward_index(node::DecisionNode) = getfield(node, :reward_index)

AbstractTrees.NodeType(::Type{<:DecisionNode}) = HasNodeType()
AbstractTrees.nodetype(::Type{<:DecisionNode{N}}) where N = DecisionNode{N}

function Base.getproperty(node::DecisionNode, value::Symbol)
    return _getprop(node, getfield(node, :information), value)
end

@generated function _getprop(node::DecisionNode, nt::NamedTuple{fields}, value::Symbol) where fields
    :(
        value ∈ fields ? getfield(nt, value) : getfield(node, value)
    )
end
 
function add_child!(node::DecisionNode{N}, child::DecisionNode{N}) where N
    any(==(child.action), map(get_action, children(node))) && return node
    push!(node.children, child)
    return node
end

function add_child!(node::DecisionNode, action::Int; reward_index::Int = 0, kwargs...)
    child = DecisionNode(action, DecisionNode[]; reward_index, kwargs...)
    add_child!(node, child)
end

function Base.in(actions::Vector{Int}, node::DecisionNode)
    current = node
    @inbounds for i in axes(actions, 1)
        action = actions[i]
        id = isempty(children(current)) ? nothing : findfirst(==(action), map(get_action, children(current)))
        isnothing(id) && return false
        current = children(current)[id]
    end
    return true
end

function Base.get!(f::Function, node::DecisionNode{N}, action::Int) where N
    @assert first(Base.return_types(f)) == N "The return type of $f is not compatible with the node information $N"
    if action ∉ map(get_action, children(node))
        child = DecisionNode(action, DecisionNode[], f(), 0)::DecisionNode{N}
        push!(children(node), child)
        return get_information(child)::N
    end
    return get_information(node)::N
end

function Base.get!(f::Function, node::DecisionNode{N}, actions::Vector{Int}) where N
    @assert first(Base.return_types(f)) == N "The return type of $f is not compatible with the node information $N"
    current = node
    @inbounds for i in axes(actions[1:end-1],1)
        action = actions[i]
        id = isempty(children(current)) ? nothing : findfirst(==(action), map(get_action, children(current)))
        isnothing(id) && return get_information(node)::N
        current = children(current)[id]
    end
    get!(f, current, actions[end])::N
end

function get_node(node::DecisionNode, actions::Vector{Int})
    current = node
    @inbounds for i in axes(actions, 1)
        action = actions[i]
        id = isempty(children(current)) ? nothing : findfirst(==(action), map(get_action, children(current)))
        isnothing(id) && return nothing
        current = children(current)[id]
    end
    current
end


struct DecisionTree{T, N}
    "The root of the decision tree"
    root::DecisionNode{N}
    "The rewards associated with the current action"
    rewards::Vector{T}
end

DecisionTree{T}(root::DecisionNode{N}) where {T, N} = DecisionTree{T, N}(root, T[])

Base.in(actions::Vector{Int}, tree::DecisionTree) = Base.in(actions, tree.root)

function Base.get!(f::Function, tree::DecisionTree, action::A)
    
end
