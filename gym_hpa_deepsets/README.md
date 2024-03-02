# DeepSets in gym-hpa

This is an extension of the original gym-hpa framework, where we attempt to implement a <b>DeepSets neural network</b> to train our agent. This is a neural network architecture promising to be scalable, meaning that you can train it in a smaller, less complex and computationally intensive environment and then apply it in a different, bigger one, without having to retrain

## How does it work?
The network takes as input a tensor, containing the normalized observation per microservice and a number, corresponding to the action the agent will make. It is of size:
<br><br>(#microserves * #actions, #metrics/microservice + 1)<br><br>
Then, the agent produces in its output the logits, a number indicating the probability of choosing each tuple/action. 
<br>During training, the agent chooses an action randomly, but taking into account the logits probability, and during testing it always chooses the action of the highest probability.
<br>This way, the agent chooses a tuple, and the number of its order corresponds to the action (e.g. 17). Then, this number is converted to a multi discrete equivalent, so as not to change the the redis implementation of the action space. 
<br>In the example’s case, it becomes (1,2), meaning in the second application, add 2.

## What made it work?
Normalizing the environment by dividing each observation by its max value appears to be essential for DS network to work. Without it, the agent is incapable of learning correctly, and apparently spams the none action, even when the penalty is on.
<br><br>Also, the metrics used in the observation space are a subset of the entire span. We only used current pods, cpu and memory utilization. Which is impressive, considering that expected pods isn’t provided to the agent
<br><br>Finally, the environment is of Subprocess Vectorized type, meaning that we execute multiple environment instances at the same time, which accelerates the training time. (Do not use it cluster mode, as there we only have one possible environment)



## Team

* [Jose Santos](https://scholar.google.com/citations?hl=en&user=57EIYWcAAAAJ)

* [Tim Wauters](https://scholar.google.com/citations?hl=en&user=Kvxp9iYAAAAJ)

* [Bruno Volckaert](https://scholar.google.com/citations?hl=en&user=NIILGOMAAAAJ)

* [Filip de Turck](https://scholar.google.com/citations?hl=en&user=-HXXnmEAAAAJ)

## Contact

If you want to contribute, please contact:

Lead developer: [Jose Santos](https://github.com/jpedro1992/)

For questions or support, please use GitHub's issue system.

## License

Copyright (c) 2020 Ghent University and IMEC vzw.

Address: IDLab, Ghent University, iGent Toren, Technologiepark-Zwijnaarde 126 B-9052 Gent, Belgium 

Email: info@imec.be.


