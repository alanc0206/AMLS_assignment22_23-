import A1.A1_code
import A2.A2_code
import B1.B1_code
import B2.B2_code

while True:
    Input = input('Enter a integer to select the model: 1.A1 2.A2 3.B1 4.B2\n')
    if Input == str(1):
        A1.A1_code.main()
    elif Input == str(2):
        A2.A2_code.main()
    elif Input == str(3):
        B1.B1_code.main()
    elif Input == str(4):
        B2.B2_code.main()
    else:
        print('Input invalid')