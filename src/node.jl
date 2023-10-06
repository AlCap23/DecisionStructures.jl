"""
    $(TYPEDEF)

    The basic structure of the decision node.

    ## Fields

    $(FIELDS)

    ## Additional Information

    Each node has the potential to store additional information in form of a `NamedTuple`. 
    The choice of the additional information should be consistent over all `DecisionNode`s of a tree. 
    Additional information can be added either via the keywords in the constructor or by constructing a `DecisionNode`
    and passing in the `NamedTuple`. 
    All information inside the `NamedTuple` can be acceseed via the `getproperty` overload, allowing the syntax

    ```julia
    node = DecisionNode(0; a = 3, b = "Info" , something = nothing)
    node.a # Returns 3
    node.b # Returns "Info"
    node.something # Returns nothing
    ```
"""
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

"""
    $(SIGNATURES)

    Return the `action` of the given `node`.
"""
get_action(node::DecisionNode) = getfield(node, :action)


"""
    $(SIGNATURES)

    Return the `information` of the given `node`.
"""
get_information(node::DecisionNode) = getfield(node, :information)


"""
    $(SIGNATURES)

    Return the `reward_index` of the given `node`.
"""
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


"""
    $(SIGNATURES)

    Add a `DecisionNode` as the direct descendant of the `node`.
"""
function add_child!(node::DecisionNode{N}, child::DecisionNode{N}) where N
    any(==(child.action), map(get_action, children(node))) && return node
    push!(node.children, child)
    return node
end

function add_child!(node::DecisionNode, action::Int; reward_index::Int = 0, kwargs...)
    child = DecisionNode(action, DecisionNode[]; reward_index, kwargs...)
    add_child!(node, child)
end

"""
    $(SIGNATURES)

    Checks if, starting from the provided `node`, the sequence of `actions` can be found.
"""
function Base.in(actions::AbstractVector{Int}, node::DecisionNode)
    current = node
    @inbounds for i in axes(actions, 1)
        action = actions[i]
        id = isempty(children(current)) ? nothing : findfirst(==(action), map(get_action, children(current)))
        isnothing(id) && return false
        current = children(current)[id]
    end
    return true
end


"""
    $(SIGNATURES)

    Return the node information if `node` has a `child` with the corresponding `action`.
    If no `action` is present, store `key => f()` as a new child. 
    
    This is intended to be called using `do` block syntax.
"""
function Base.get!(f::Function, node::DecisionNode{N}, action::Int) where N
    @assert first(Base.return_types(f)) == N "The return type of $f is not compatible with the node information $N"
    id = isempty(children(node)) ? nothing : findfirst(==(action), map(get_action, children(node)))
    if isnothing(id)
        child = DecisionNode(action, DecisionNode[], f(), 0)::DecisionNode{N}
        push!(children(node), child)
        return get_information(child)::N
    end
    node = getindex(children(node), id)
    return get_information(node)::N
end

"""
    $(SIGNATURES)

    Return the node information if a `DecisionNode` can be reached using the sequence of `actions` up until the 
    last one. If for any action no corresponding descendent can be found, the `information` of the initial `node` is
    returned.
    
    This is intended to be called using `do` block syntax.
"""
function strict_get!(f::Function, node::DecisionNode{N}, actions::AbstractVector{Int}) where N
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

"""
    $(SIGNATURES)

    Return the node information if a `DecisionNode` can be reached using the sequence of `actions` up until the 
    last one. If for any action no corresponding descendent can be found, a new one is created using the same informatio
    provided by `f()`.
    
    This is intended to be called using `do` block syntax.
"""
function Base.get!(f::Function, node::DecisionNode{N}, actions::AbstractVector{Int}) where N
    @assert first(Base.return_types(f)) == N "The return type of $f is not compatible with the node information $N"
    current = node
    information = f()
    @inbounds for i in axes(actions,1)
        action = actions[i]
        id = isempty(children(current)) ? nothing : findfirst(==(action), map(get_action, children(current)))
        if isnothing(id) 
            child = DecisionNode(action; information...)
            push!(children(current), child)
            current = child
        else
            current = children(current)[id]
        end
    end
    return information
end

"""
    $(SIGNATURES)

    Find the descendent of `node` corresponding to the provided sequence of `actions` starting
    from `node`. If no descendent is found, returns `nothing`.
"""
function get_node(node::DecisionNode, actions::AbstractVector{Int})
    isempty(actions) && return node
    current = node
    @inbounds for i in axes(actions, 1)
        action = actions[i]
        id = isempty(children(current)) ? nothing : findfirst(==(action), map(get_action, children(current)))
        isnothing(id) && return nothing
        current = children(current)[id]
    end
    current
end

