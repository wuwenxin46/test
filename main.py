# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


# def print_hi(name):
#     # 在下面的代码行中使用断点来调试脚本。
#     print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。
#
#
# # 按间距中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助

a = [1.2616434,2.1248896,1.2538548,0.89937556,0.91424096,92.9112,1.015598,0.0054545826,0.7892291,0.3264987,
     0.35822225,0.16315638,0.97250795,0.94249105,8.22814e-06,0.016610045,0.041091442,94.04382,1.1725894,0.024940148,
     0.031851396,0.011731685,0.09909733,0.29393673,0.20179574,0.11531567,0.46853042,0.17425142,0.25815305,0.116440214,
     88.731316,0.41260988,0.9670624,1.0644966,0.16708852,0.7819508,0.20719509,0.000544214,0.21239369,0.021362202,
     0.77388424,0.91267216,0.3249621,1.286567,85.69577,0.2593328,0.19131559,0.081201114,0.30420974,0.64052665,
     0.2888736,0.75191873,85.61215,0.94029564,0.871555,0.7276645,0.6753171,0.59240955,0.25001055,0.78603804,
     0.73734766,0.885002,0.40007177,0.5488751,0.057741225,0.6255114]
sum = 0
for i in range(len(a)):
    sum += a[i]
b = sum/len(a)
print(b)
