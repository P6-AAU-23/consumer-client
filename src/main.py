from src.controller import Controller


def main(args: any) -> None:
    controller = Controller(args)
    controller.run()
