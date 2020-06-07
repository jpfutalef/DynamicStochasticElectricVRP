# GA for the E-VRP-BD-NL-CCS

## About
Here we present a Python GA implementation that's been able to solve E-VRP 
instances up to 100 customers using 5 EVs. We consider an E-VRP with:
 - battery degradation, 
 - non-linear charging functions,
 - capacitated CS,
 - customer time windows,
 - variable travel times,
 - EV limitations.
 
The implementations allows you to define fleets and network independently; 
thus, different fleets can be tested to solve the same problem and provide 
recommendations on the best fleet characteristics (size and EV specifications).
 
## Workflow

The whole operation is divided into two main stages:
1. Pre-operation: Assignation of customers and initial routes, before EVs
begin the operation.
2. On-line operation: Improvement of routes based on real-time measurement 
of travel times and traffic state.

## Usage
### Pre-operation

### On-line operation

 



