using DecisionStructures
using AbstractTrees
using Test


@testset "DecisionStructures.jl" begin

    @testset "Nodes" begin 
        f = () -> (; a = rand(Int64), b = reduce(*,rand(Char, 4)), c = rand(Float64, 2))
        g = () -> (; a = rand(Int64), b = rand(Int64), c = rand(Float64, 2))
    
        root = DecisionNode(0; f()...)
        @test get_action(root) == 0
        @test get_reward_index(root) == 0 
        @test isa(get_information(root), NamedTuple) 
    
        @test_throws MethodError add_child!(root, 0; g()...)
        @test_nowarn add_child!(root, 1; f()...)
        @test treebreadth(root) == 1
        @test_nowarn add_child!(root, 1; f()...)
        @test treebreadth(root) == 1
        @test [1] ∈ root
        @test [1, 3] ∉ root
        @test_nowarn get!(f, root, [1,3])
        @test treeheight(root) == 2
        @test_nowarn get!(f, root, 2)
        @test_nowarn get!(f, root, [2,3])
        @test treebreadth(root) == 2
        @test isa(get_node(root, [2,3]), typeof(root))
        @test_throws AssertionError get!(g, root, [2,4])
    end


    @testset "Tree" begin 

        f = () -> (; a = rand(Int64), b = reduce(*,rand(Char, 4)), c = rand(Float64, 2))
        g = () -> (; a = rand(Int64), b = rand(Int64), c = rand(Float64, 2))
        reward = (actions::Vector{Int}) -> argmax(actions) - argmin(actions)

        root = DecisionNode(0; f()...)
        tree = DecisionTree(root, reward)
        @test_nowarn get!(f, tree, [1])
        @test_nowarn get!(f, tree, [1,2])
        @test_nowarn get!(f, tree, [1,2,3])
        @test_nowarn get!(f, tree, [[1,2,3], [1,2,4]])

    end
    
end


