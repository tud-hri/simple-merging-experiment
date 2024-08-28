---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
---


<h1> Welcome </h1>
This web page contains the supplementary material for the paper <a href="https://doi.org/10.1109/OJITS.2024.3349635" target="_blank">Human Merging Behaviour in a Coupled Driving Simulator: How do we resolve conflicts?</a> by Olger Siebinga, Arkady Zgonnikov, and David A. Abbink. You can find the source code to reproduce the experiments and the plots on <a href="https://github.com/tud-hri/simple-merging-experiment" target="_blank">Github</a>. You can find the data from the paper on the <a href="https://doi.org/10.4121/19550377.v1" target="_blank">4TU Data repository</a>.

In all the plots below, you can click the legend to toggle elements in the plot. Other navigation tools (like zoom and reset view) can be found in the top right corner of every frame.

<h2>Who Merged First?</h2>
<h3>In 2D</h3>
<iframe src="plots/who_first_box.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h3>In 3D</h3>
<iframe src="plots/surf.html" style="width:100%;aspect-ratio: 2/1;"></iframe>

<h2>Velocity traces plot</h2>
The plots below show all velocity traces for all trials, sorted per condition. Left en right correspond to a single participant in a pair. Click on the legend entries to enable or disable specific traces. 

<h4>Condition -4_-8</h4>
<iframe src="plots/-4_-8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition -4_0</h4>
<iframe src="plots/-4_0.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition -4_8</h4>
<iframe src="plots/-4_8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition -2_8</h4>
<iframe src="plots/-2_8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 0_8</h4>
<iframe src="plots/0_8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 0_0</h4>
<iframe src="plots/0_0.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 0_-8</h4>
<iframe src="plots/0_-8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 2_-8</h4>
<iframe src="plots/2_-8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 4_-8</h4>
<iframe src="plots/4_-8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 4_0</h4>
<iframe src="plots/4_0.html" style="width:100%;aspect-ratio: 2/1;"></iframe>
<h4>Condition 4_8</h4>
<iframe src="plots/4_8.html" style="width:100%;aspect-ratio: 2/1;"></iframe>

<h2>Initial Actions</h2>
<iframe src="plots/initial_action.html" style="width:100%;aspect-ratio: 2/1;"></iframe>

<h2>CRT over Conditions</h2>
<iframe src="plots/3d_boxplot.html" style="width:100%;aspect-ratio: 2/1;"></iframe>

<h2>Learning effect during the experiment</h2>
To check if learning effect were present during the experiment, we investigated if the performance and effort of the drivers changes over the trials. For performance, we used the collisions as a metric, for effort we used the maximum deviation from the initial velocity. Based on the plot below, we concluded that there is no evidence that suggests leaning effects were present after the training phase.

![Collisions over trials](plots/collisions_over_trials.png)
![Effort over trials](plots/actions_over_trials.png)
