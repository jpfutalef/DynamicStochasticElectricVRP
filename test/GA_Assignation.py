from res.GA_Assignation import *
from models.Fleet import from_xml, InitialCondition

if __name__ == '__main__':
    m = 2
    r = 2
    num_cust = 5
    num_cs = 2
    st_points = {0: InitialCondition(0, 0, 1., 80., 0.5), 1: InitialCondition(0, 0, 1., 80., 0.5)}

    individual1 = [3, 2, 4, '|', 5, 1, '|', 4, 6, 10., -1, 6, 13., -1, 7, 15., -1, 6, 15., 60 * 9, 60 * 10]
    individual2 = [4, 1, 5, 3, '|', 2, '|', 4, 7, 10., -1, 6, 13., -1, 7, 15., -1, 6, 15., 60 * 9, 60 * 10]
    individual3 = [4, 1, '|', 5, 2, 3, '|', -1, 6, 10., 2, 6, 13., -1, 7, 15., -1, 7, 15., 60 * 9, 60 * 10]

    print('  ***** MUTATION *****')
    count = 0
    while count < 100:
        for i in range(21):
            print(f'Individual 1:\n {individual1}\nMutated:\n {mutate(individual1, m, num_cust, num_cs, r, int(i))}')
        count += 1

    print('  ***** CROSSOVER *****')
    count = 0
    while count < 100:
        for i in range(21):
            print('  Original....')
            print(f'  Individual 1:\n {individual1}')
            print(f'  Individual 2:\n {individual2}')
            print('  Crossover....')
            crossover(individual1, individual2, m, r, int(i))
            print(f'  Individual 1:\n {individual1}')
            print(f'  Individual 2:\n {individual2}')
            print()
        count += 1

    print('  ***** RANDOM INDIVIDUALS *****')
    while True:
        i1 = input('index: ')
        if i1 == 's':
            break
        print(f'  New individual:\n {random_individual(num_cust, num_cs, m, r)}')
    a = 1
