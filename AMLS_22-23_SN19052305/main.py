import A1.A1_code
import A2.A2_code
import B1.B1_code
import B2.B2_code

while True:
    Input = input('Enter a integer to select the model: 0:Exit 1.A1 2.A2 3.B1 4.B2\n')
    if Input == str(0):
        print('Exit')
        break
    elif Input == str(1):
        print('Running A1_code')
        A1.A1_code.main()
    elif Input == str(2):
        print('Running A2_code')
        A2.A2_code.main()
    elif Input == str(3):
        print('Running B1_code')
        B1.B1_code.main()
    elif Input == str(4):
        print('Running B2_code')
        B2.B2_code.main()
    else:
        print('Input invalid')
