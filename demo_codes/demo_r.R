install.packages("igraph")

# Load thư viện (sửa lỗi chính tả)
library(igraph)

# Tạo một đồ thị
g <- graph_from_literal(A-B, A-C, B-C, C-D, D-E)

# Tính toán độ trung tâm
closeness <- closeness(g)
betweenness <- betweenness(g)

print(closeness)
print(betweenness)

# Vẽ đồ thị
plot(g)
