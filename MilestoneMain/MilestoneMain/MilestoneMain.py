
# main.py

def main():
    print("Seleccione el script que desea ejecutar:")
    print("1. milestone1")
    print("2. milestone2")
    print("3. milestone3")
    print("4. milestone4")
    print("5. milestone5")
    print("6. milestone6")
    print("7. milestone7")

    choice = input("Ingrese el numero del script: ")

    if choice == "1":
        import milestone1
        milestone1.run_script()
    elif choice == "2":
        import milestone2_modulos
        milestone2_modulos.run_script()
    elif choice == "3":
        import Milestone3
        Milestone3.run_script()
    elif choice == "4":
        import Milestone4
        Milestone4.run_script()
    elif choice == "5":
        import Milestone5_Nbody
        Milestone5_Nbody.run_script()
    elif choice == "6":
        import Milestone_6
        Milestone5_Nbody.run_script()
    elif choice == "7":
        import Milestone7
        Milestone7.run_script()
    else:
        print("Opcion no valida.")

if __name__ == "__main__":
    main()