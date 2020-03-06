                               第三周作业
1.运行
利用了pyspherepack模块实现了相同圆包装在正方形内的功能
通过pyspherepack中的setup安装，运行python setup.py install.
然后运行packag.py即可，可以在程序中改变球的个数、正方形的边界、以及迭代的次数（越大越精确）

2.结果
参考数值：
http://hydra.nat.uni-magdeburg.de/packing/csq/csq.html
https://en.wikipedia.org/wiki/Circle_packing_in_a_square

       Density	packag.py	参考数值
	2 balls	53.90	       53.90
       5 balls	67.37	       67.37
       10 balls	67.95	       69.00
       20 balls	71.92	       77.95

由于md文件暂防不了结果图，结果图已上传至github。
Density即所有圆面积与正方形面积的比值

3.说明
程序原因导致正方形的坐标并不是严格的-1到1，但是只需要简单的缩放即可，在计算圆面积与正方形面积之比并不会受到影响，可以根据此来验证答案。

可以计算出圆的半径r=r'*2/l，l为程序输出正方形的长，r'为原本程序中计算出的r。在结果图中已经将此过程完成，见fig_result黄色高亮部分。

另外当圆个数为1的时候会有问题，个数应该大于等于2.
