from res.GA_Assignation import *

if __name__ == '__main__':
    m = 2
    r = 2
    num_cust = 5
    num_cs = 2
    st_points = {0: InitialCondition(0, 0, 1., 80., 0.5), 1: InitialCondition(0, 0, 1., 80., 0.5)}

    individual1 = [3, 2, 4, '|', 5, 1, '|', 4, 6, 10., -1, 6, 13., -1, 7, 15., -1, 6, 15., 60 * 9, 60 * 10]
    individual2 = [4, 1, 5, '|', '|', 2, 3, '|', 4, 7, 10., -1, 6, 13., -1, 7, 15., -1, 6, 15., 60 * 9, 60 * 10]
    individual3 = ['|', 1, '|', 4, 5, 2, 3, '|', -1, 6, 10., 2, 6, 13., -1, 7, 15., -1, 7, 15., 60 * 9.5, 60 * 10.5]
    inds = [individual1, individual2, individual3]

    for i in range(0):
        ind = int(input('individual?: '))
        index = int(input('index?:    '))
        customer_block_indices(inds[ind], index)

    print('  ***** RANDOM INDIVIDUALS *****')
    for i in range(0):
        print(f'  New individual:\n {random_individual(num_cust, num_cs, m, r)}')

    print('  ***** MUTATION *****')
    for i in range(0):
        print(f'Individual:\n  {individual1}\nMutated:\n  {mutate_charging_operation1(individual1, int(i), m, num_cust, num_cs, r)}')

    print('  ***** CROSSOVER *****')
    for i in range(len(individual2)):
        print('  Original....')
        print(f'  Individual 1:\n {individual2}')
        print(f'  Individual 2:\n {individual3}')
        print('  Crossover....')
        crossover(individual2, individual3, m, r, int(i))
        print(f'  Individual 1:\n {individual2}')
        print(f'  Individual 2:\n {individual3}')
        print()
