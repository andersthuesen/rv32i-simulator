#!/usr/bin/env python3
import os
import sys
from enum import Enum
import struct
import binascii

class Operation(Enum):
  LUI = 0b0110111  
  AUIPC = 0b0010111
  JAL = 0b1101111
  JALR = 0b1100111
  BRANCH = 0b1100011
  LOAD = 0b0000011
  STORE = 0b0100011
  IMM = 0b0010011
  ARITH = 0b0110011
  FENCE = 0b0001111
  SYSTEM = 0b1110011

class Funct3(Enum):
  ADD = SUB = ADDI = 0b000
  SLL = SLLI = 0b001
  SLT = SLTI = 0b010
  SLTU = SLTIU = 0b011

  XOR = XORI = 0b100
  SRL = SRLI = SRA = SRAI = 0b101
  OR = ORI = 0b110
  AND = ANDI = 0b111

  BEQ = 0b000
  BNE = 0b001
  BLT = 0b100
  BGE = 0b101
  BLTU = 0b110
  BGEU = 0b111

  LB = SB = 0b000
  LH = SH = 0b001
  LW = SW = 0b010
  LBU = 0b100
  LHU = 0b101

class Funct7(Enum):
  DEFAULT = 0b0000000 # Default
  ALTERNATIVE = 0b0100000 # Alternative

class Bits:
  def __init__(self, value: bytes):
    self.value = value
  
  def __getitem__(self, index):
    if isinstance(index, slice):
      return (self.value >> index.stop) & ((1 << (index.start - index.stop + 1)) - 1)
    else:
      return (self.value >> index) & 1

  def __str__(self) -> str:
    return f"{self.value:32b}"


class Registers:
  def __init__(self):
    self.regs = [0] * 32

  def __getitem__(self, index):
    return self.regs[index]

  def __setitem__(self, index, value):
    if index != 0: # Don't write to x0.
      # Keeps values 32-bit
      self.regs[index] = value & 0xFFFFFFFF

  def __str__(self) -> str:
    return str(self.regs)


def decode_instr(instr: int):
  bits = Bits(instr) # Make it easier to index bits.
  op = Operation(bits[6:0])
  funct3 = Funct3(bits[14:12])
  funct7 = bits[31:25]
  rd = bits[11:7]
  rs1 = bits[19:15]
  rs2 = bits[24:20]
  imm_i = sign(bits[31:20], 12)
  imm_s = sign(bits[31:25] << 5 | bits[11:7], 12)
  imm_b = sign(bits[31] << 12 | bits[30:25] << 5 | bits[11:8] << 1 | bits[7:7] << 11, 13)
  imm_u = sign(bits[31:12] << 12, 32)
  imm_j = sign(bits[31] << 20 | bits[30:21] << 1 | bits[21:20] << 11 | bits[19:12] << 12, 21)
  
  return op, funct3, funct7, rd, rs1, rs2, imm_i, imm_s, imm_b, imm_u, imm_j

def sign(value: int, num_bits: int) -> int:
  """
  Signs an input integer.
  """
  if value >> (num_bits - 1) == 1:
    return -((1 << num_bits) - value)
  else:
    return value

class Memory:
  def __init__(self, mem_size: int, initial: bytes = b''):
    self.mem = bytearray(mem_size)
    self.mem[0:len(initial)] = initial

  def __getitem__(self, address: int):
    return self.read_byte(address)

  def read_byte(self, address: int):
    return self.mem[address]

  def read_halfword(self, address: int) -> int:
    return struct.unpack('<H', self.mem[address:address+2])[0]

  def read_word(self, address: int) -> int:
    return struct.unpack('<I', self.mem[address:address+4])[0]

  def write_byte(self, address: int, byte: int):
    self.mem[address] = byte

  def write_halfword(self, address: int, halfword: int):
    self.mem[address:address+2] = struct.pack('<H', halfword)
  
  def write_word(self, address: int, word: int):
    self.mem[address:address+4] = struct.pack('<I', word)

  def __setitem__(self, address: int, byte: int):
    self.write_byte(address, byte)

  def __str__(self) -> str:
    return str(self.mem)

class RV32ICore:
  def __init__(self, memory: bytes):
    # Initialize 32 registers, plus program counter to 0.
    self.regs = Registers()
    self.pc = 0
    self.memory = Memory(mem_size=0x100000, initial=memory) # 1MB of memory

  def execute(self):
    while True:
      instr =  self.memory.read_word(self.pc)
      op, funct3, funct7, rd, rs1, rs2, imm_i, imm_s, imm_b, imm_u, imm_j = decode_instr(instr)

      if op == Operation.LUI:
        self.regs[rd] = imm_u

      elif op == Operation.AUIPC:
        self.regs[rd] = self.pc + imm_u

      elif op == Operation.ARITH:
        if funct3 == Funct3.ADD and Funct7(funct7) == Funct7.DEFAULT:
          self.regs[rd] = self.regs[rs1] + self.regs[rs2]
        elif funct3 == Funct3.SUB and Funct7(funct7) == Funct7.ALTERNATIVE:
          self.regs[rd] = self.regs[rs1] - self.regs[rs2]
        elif funct3 == Funct3.SLL:
          self.regs[rd] = self.regs[rs1] << self.regs[rs2]
        elif funct3 == Funct3.SLT:
          self.regs[rd] = int(sign(self.regs[rs1], 32) < sign(self.regs[rs2], 32))
        elif funct3 == Funct3.SLTU:
          self.regs[rd] = int(self.regs[rs1] & 0xFFFFFFFF < self.regs[rs2] & 0xFFFFFFFF)
        elif funct3 == Funct3.XOR:
          self.regs[rd] = self.regs[rs1] ^ self.regs[rs2]
        elif funct3 == Funct3.SRL and Funct7(funct7) == Funct7.DEFAULT:
          self.regs[rd] = self.regs[rs1] >> self.regs[rs2]
        elif funct3 == Funct3.SRA and Funct7(funct7) == Funct7.ALTERNATIVE:
          signed = self.regs[rs1] >> 31 # 1 if negativ, 0 otherwise
          # Perform logical shift
          self.regs[rd] = self.regs[rs1] >> (self.regs[rs2] & 0x1F) 
          # If the sign bit was set, set signed bit again.
          self.regs[rd] |= (0xFFFFFFFF * signed) << (32 - (self.regs[rs2] & 0x1F))
        elif funct3 == Funct3.OR:
          self.regs[rd] = self.regs[rs1] | self.regs[rs2]
        elif funct3 == Funct3.AND:
          self.regs[rd] = self.regs[rs1] & self.regs[rs2]
        else:
          raise Exception(f"Unknown arithmetic operation {op} {funct3}")

      elif op == Operation.IMM:
        if funct3 == Funct3.ADDI:
          self.regs[rd] = self.regs[rs1] + imm_i
        elif funct3 == Funct3.SLLI:
          self.regs[rd] = self.regs[rs1] << imm_i
        elif funct3 == Funct3.SLTI:
          self.regs[rd] = int(sign(self.regs[rs1], 32) < sign(imm_i, 32))
        elif funct3 == Funct3.SLTIU:
          self.regs[rd] = int(self.regs[rs1] & 0xFFFFFFFF < imm_i & 0xFFFFFFFF)
        elif funct3 == Funct3.XORI:
          self.regs[rd] = self.regs[rs1] ^ imm_i
        elif funct3 == Funct3.SRLI and Funct7(funct7) == Funct7.DEFAULT:
          self.regs[rd] = self.regs[rs1] >> imm_i
        elif funct3 == Funct3.SRAI and Funct7(funct7) == Funct7.ALTERNATIVE:
          signed = self.regs[rs1] >> 31 # 1 if negativ, 0 otherwise
          # Perform logical shift
          self.regs[rd] = self.regs[rs1] >> (imm_i & 0x1F) 
          # If the sign bit was set, set signed bit again.
          self.regs[rd] |= (0xFFFFFFFF * signed) << (32 - (imm_i & 0x1F))
        elif funct3 == Funct3.ORI:
          self.regs[rd] = self.regs[rs1] | imm_i
        elif funct3 == Funct3.ANDI:
          self.regs[rd] = self.regs[rs1] & imm_i
        else:
          raise Exception("Unsupported immediate instruction: {op} {funct3}")

      elif op == Operation.SYSTEM:
        output = bytes()
        for reg in self.regs.regs:
          output += struct.pack("<I", reg)
        return output
        
      elif op == Operation.BRANCH:
        if funct3 == Funct3.BEQ:
          if self.regs[rs1] == self.regs[rs2]:
            self.pc = self.pc + imm_b
            continue
        elif funct3 == Funct3.BNE:
          if self.regs[rs1] != self.regs[rs2]:
            self.pc = self.pc + imm_b
            continue
        elif funct3 == Funct3.BLT:
          if sign(self.regs[rs1], 32) < sign(self.regs[rs2], 32):
            self.pc = self.pc + imm_b
            continue
        elif funct3 == Funct3.BGE:
          if sign(self.regs[rs1], 32) >= sign(self.regs[rs2], 32):
            self.pc = self.pc + imm_b
            continue
        elif funct3 == Funct3.BLTU:
          if self.regs[rs1] & 0xFFFFFFFF < self.regs[rs2] & 0xFFFFFFFF:
            self.pc = self.pc + imm_b
            continue
        elif funct3 == Funct3.BGEU:
          if self.regs[rs1] & 0xFFFFFFFF >= self.regs[rs2] & 0xFFFFFFFF:
            self.pc = self.pc + imm_b
            continue
        else:
          raise Exception(f"Unknown branch operation {op} {funct3}")
            
      elif op == Operation.STORE:
        if funct3 == Funct3.SB:
          self.memory.write_byte(self.regs[rs1] + imm_s, self.regs[rs2] & 0xFF)
        elif funct3 == Funct3.SH:
          self.memory.write_halfword(self.regs[rs1] + imm_s, self.regs[rs2] & 0xFFFF)
        elif funct3 == Funct3.SW:
          self.memory.write_word(self.regs[rs1] + imm_s, self.regs[rs2])
        else:
          raise Exception("Unsupported store instruction: {op} {funct3}")

      elif op == Operation.LOAD:
        if funct3 == Funct3.LB:
          self.regs[rd] = sign(self.memory.read_byte(self.regs[rs1] + imm_i), 8)
        elif funct3 == Funct3.LH:
          self.regs[rd] = sign(self.memory.read_halfword(self.regs[rs1] + imm_i), 16)
        elif funct3 == Funct3.LW:
          self.regs[rd] = sign(self.memory.read_word(self.regs[rs1] + imm_i), 32)
        elif funct3 == Funct3.LBU:
          self.regs[rd] = self.memory.read_byte(self.regs[rs1] + imm_i) & 0xFF
        elif funct3 == Funct3.LHU:
          self.regs[rd] = self.memory.read_halfword(self.regs[rs1] + imm_i) & 0xFFFF
        else:
          raise Exception("Unsupported load instruction: {op} {funct3}")

      elif op == Operation.JAL:
        self.regs[rd] = self.pc + 4
        self.pc = self.pc + imm_j
        continue

      elif op == Operation.JALR:
        self.regs[rd] = self.pc + 4
        self.pc = self.regs[rs1] + imm_i
        continue

      else:
        raise Exception(f"Unknown operation {op} {funct3}")

      self.pc += 4

  def __call__(self):
    return self.execute()


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: ./sim.py <tests path>")
    sys.exit(1)
  

  tests_path = sys.argv[1]
  for task in ["task1", "task2", "task3", "task4"]:
    print("=" * 16 + f" {task} " + "=" * 16)
    task_path = os.path.join(tests_path, task)
    files = os.listdir(task_path)
    for file in filter(lambda f: f.endswith(".bin"), files):
      print(file, end=" " * (16 - len(file)))
      
      with open(os.path.join(task_path, file), "rb") as program_file:
        program = program_file.read()
        core = RV32ICore(program)
        output = core.execute()
        with open(os.path.join(task_path, file.replace(".bin", ".res")), "rb") as result_file:
          result = result_file.read()
          if output == result[0:len(output)]:
            print("PASSED")
          else:
            print("FAILED")
            print("Got:")
            print(binascii.hexlify(output))
            print("Excepted:")
            print(binascii.hexlify(result))


  
