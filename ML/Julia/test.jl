# using ScikitLearn, DecisionTree

labels = []
values = []

open("../data/mnist_test.csv") do file
    for ln in eachline(file)

        data = map(x->parse(Float64,x), split(ln, ","))
        append!(labels, data[1])
        push!(values, data[2:size(data)[1]])
    end
end

println(size(labels)[1])
println(labels[1])
for i in 1:28
    for j in 1:28
        print(values[1][28 * (i - 1) + j] < 0.001 ? 0 : 1)
    end
    println("")
end