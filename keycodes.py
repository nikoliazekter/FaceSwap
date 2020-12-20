from string import ascii_uppercase, digits


class Keycode:
    # platform-dependent, these are for Windows
    ARROW_LEFT = {2424832}
    ARROW_UP = {2490368}
    ARROW_RIGHT = {2555904}
    ARROW_DOWN = {2621440}


# add class attributes for ASCII letters
for c in ascii_uppercase:
    setattr(Keycode, c, {ord(c), ord(c.lower())})
# add class attributes for digits
for c in digits:
    setattr(Keycode, f'NUM{c}', {ord(c)})
