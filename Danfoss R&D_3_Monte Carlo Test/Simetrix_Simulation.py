import csv
import os
from distutils.dir_util import copy_tree
from shutil import copytree, ignore_patterns, rmtree, copyfile

class SimetrixSimulation:
    @property
    def module(self):
        return self._module

    @property
    def moduleCircuitBlock(self):
        return self._moduleCircuitBlock

    @property
    def simetrixFolder(self):
        return self._simetrixFolder

    @property
    def schematics(self):
        return self._schematics

    @property
    def templateFolder(self):
        return self._templateFolder

    @property
    def solver(self):
        return self._solver

    @property
    def deviceName(self):
        return self._deviceName

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def chip(self):
        return self._chip

    def __init__(self, module, moduleCircuitBlock, ansysFolder, schematics=None, templateFolder=None, params=None,
                 solver=None, chip=None, simetrixFolder=None):
        self._params = params
        if not "T_Ambient" in self.params:
            self.params.update({"T_Ambient": 25})

        if not "Rmc_HS" in self.params:
            self.params.update({"Rmc_HS": 1e-3})

        if not "Rmc_LS" in self.params:
            self.params.update({"Rmc_LS": 1e-3})

        if not "L_gate" in self.params:
            self.params.update({"L_gate": 2e-9})

        if not "Delta_t" in self.params:
            self.params.update({"Delta_t": "0.1n"})

        self._module = module
        self._moduleCircuitBlock = moduleCircuitBlock
        self._simetrixFolder = ansysFolder if simetrixFolder is None else simetrixFolder
        self._schematics = schematics
        self._templateFolder = templateFolder
        self._solver = 'C:\\Program Files\\SIMetrix830\\bin64\\SIMetrix.exe' if solver is None else solver
        self._deviceName = 'DUT_' + self._module.templateFolder.replace(
            os.path.dirname(self._module.templateFolder) + '\\', '')
        self._chip = chip
        if not ansysFolder == simetrixFolder:
            src = os.path.join(ansysFolder, self._deviceName + '.cir')
            dst = os.path.join(simetrixFolder, self._deviceName + '.cir')
            copyfile(src, dst)

    def __str__(self):
        return "Simetrix Schematics: %s, Device Name: %s" % ("; ".join(self._schematics), self.deviceName)

    def prepare_cir_file(self):
        temp_folder = os.path.join(self.simetrixFolder, "temp")
        copytree(self.module.templateFolder, temp_folder, ignore=ignore_patterns('*.aed*', '@*'))
        copy_tree(temp_folder, self.simetrixFolder)
        rmtree(temp_folder)

    def prepare_script(self, schematic):
        script = os.path.join(self.simetrixFolder, schematic + '.sxscr')
        pinFile = os.path.join(self.module.templateFolder, "pins.csv")
        reader = csv.reader(open(pinFile, 'r'), delimiter=',')
        pins = [x[0] for x in reader]
        currents = ' '.join(['U1#' + pin for pin in pins])
        voltages = ' '.join(['U1.' + pin for pin in pins])

        with open(script, "w") as f:
            f.write('OpenSchem ' + os.path.join(self.simetrixFolder, schematic) + '\n')
            # f.write('OpenSchem ' + schematic + '\n')

            for key, value in self.params.items():
                f.write('Let global:' + key + '=' + str(value) + '\n')

            f.write('Netlist circuit.net\n')
            f.write('Run circuit.net\n')
            outputFile = schematic.strip(".sxsch") + '_Current.csv'
            f.write('Show /file ' + outputFile + ' ' + currents + '\n')
            outputFile = schematic.strip(".sxsch") + '_Voltage.csv'
            f.write('Show /file ' + outputFile + ' ' + voltages + '\n')
            f.write('Save\n')
            f.write('Close\n')
            f.write('Quit\n')
            f.close()

    def prepare_schematics(self):
        setting_string = ".tran print_step stop_time fast_start max_step"
        settings = ".tran " + self.params["Delta_t"] + " " + str(self.params["t_end"]) + ' 0 ' + self.params["Delta_t"]

        for s in self.schematics:
            with open(os.path.join(self.templateFolder, s), 'r') as src:
                lines = src.readlines()
                lines = [x.replace(setting_string, settings) for x in lines]

                if self.chip is not None:
                    temp_folder = os.path.split(self.module.templateFolder)[0]
                    for c in self.chip:
                        src = os.path.join(temp_folder, "Chip\\" + c["library"])
                        dst = os.path.join(self.simetrixFolder, c["library"])
                        copyfile(src, dst)

                        symbol_str = "@" + c["symbol"] + "@"
                        chip_model = c["library"].split(".")[0]

                        temp_str = ".temp T_Ambient"
                        temp_value = ".temp " + str(self.params["T_Ambient"])

                        lines = [x.replace(symbol_str, chip_model) for x in lines]
                        lines = [x.replace(temp_str, temp_value) for x in lines]

            with open(os.path.join(self.simetrixFolder, s), 'w') as dst:
                dst.writelines(lines)

    def run(self):
        self.prepare_schematics()
        for schematic in self._schematics:
            self.prepare_script(schematic)
            cmd = '"' + self.solver + '" /s ' + os.path.join(self.simetrixFolder, schematic + '.sxscr')

            stream = os.popen(cmd)
            output = stream.read()
            print(output)
