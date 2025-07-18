import os
import platform
import sys
import subprocess
import asyncio
import json
from pathlib import Path
from OCR.config import Settings
from OCR.pdf_processor import PDFProcessor


class PowerPointProcessor:
    def __init__(self, pptx_path):
        """
        Initialize the PowerPointProcessor with the path to the PowerPoint file.
        
        Args:
            pptx_path: Path object or string pointing to the PowerPoint file
        """
        self.pptx_path = Path(pptx_path) if not isinstance(pptx_path, Path) else pptx_path
        self.temp_pdf_path = None
        
    def _check_requirements(self):
        """Check if the required packages are installed for the current platform"""
        system = platform.system()
        
        if system == "Windows":
            try:
                import comtypes
                return True
            except ImportError:
                print("The comtypes package is required for Windows.")
                print("Install it using: pip install comtypes")
                return False
        elif system == "Darwin":  # macOS
            if not os.path.exists("/Applications/Microsoft PowerPoint.app"):
                print("Microsoft PowerPoint is not installed on this Mac")
                return False
            return True
        elif system == "Linux":
            try:
                subprocess.check_output(["which", "libreoffice"])
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("LibreOffice is required for Linux.")
                print("Install it using: sudo apt-get install libreoffice")
                return False
        return True

    def _convert_pptx_to_pdf(self, output_folder, pdf_name=None):
        """
        Convert PowerPoint presentation to PDF on any platform.
        
        Args:
            output_folder (str): Path to the output folder for PDF
            pdf_name (str): Name of the output PDF file (optional)
            
        Returns:
            Path: Path to the generated PDF file, or None if failed
        """
        # Convert paths to absolute paths
        pptx_path = os.path.abspath(self.pptx_path)
        output_folder = os.path.abspath(output_folder)
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate PDF name if not provided
        if pdf_name is None:
            base_name = os.path.splitext(os.path.basename(pptx_path))[0]
            pdf_name = f"{base_name}.pdf"
        
        # Get the operating system
        system = platform.system()
        
        try:
            if system == "Windows":
                return self._convert_pptx_windows(pptx_path, output_folder, pdf_name)
            elif system == "Darwin":  # macOS
                return self._convert_pptx_macos(pptx_path, output_folder, pdf_name)
            elif system == "Linux":
                return self._convert_pptx_linux(pptx_path, output_folder, pdf_name)
            else:
                print(f"Unsupported operating system: {system}")
                return None
        except Exception as e:
            print(f"Error converting PowerPoint to PDF: {str(e)}")
            return None

    def _convert_pptx_windows(self, pptx_path, output_folder, pdf_name):
        """Windows conversion method using COM interface with PowerPoint"""
        try:
            from comtypes import client
            
            # Start PowerPoint
            powerpoint = client.CreateObject("PowerPoint.Application")
            powerpoint.Visible = 1
            
            # Open the presentation
            presentation = powerpoint.Presentations.Open(pptx_path)
            
            try:
                # Save as PDF (format code 32 is for PDF)
                pdf_path = os.path.join(output_folder, pdf_name)
                presentation.SaveAs(pdf_path, 32)
                print(f"Successfully converted PowerPoint to PDF: {pdf_path}")
                return Path(pdf_path)
            finally:
                # Clean up
                presentation.Close()
                powerpoint.Quit()
        except ImportError:
            print("Could not import comtypes. Install it using: pip install comtypes")
            return None
        except Exception as e:
            print(f"Windows PowerPoint to PDF conversion error: {str(e)}")
            return None

    def _convert_pptx_macos(self, pptx_path, output_folder, pdf_name):
        """macOS conversion method using AppleScript"""
        try:
            # Check if PowerPoint is installed
            if not os.path.exists("/Applications/Microsoft PowerPoint.app"):
                print("Microsoft PowerPoint is not installed on this Mac")
                return None
                
            pptx_path = pptx_path.replace("\\", "/")
            output_folder = output_folder.replace("\\", "/")
            pdf_path = os.path.join(output_folder, pdf_name).replace("\\", "/")
            
            # Create AppleScript command
            script = f'''
            tell application "Microsoft PowerPoint"
                open "{pptx_path}"
                set pres to active presentation
                
                save pres in "{pdf_path}" as save as PDF
                
                close pres saving no
                quit
            end tell
            '''
            
            # Execute AppleScript
            process = subprocess.Popen(['osascript'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            process.communicate(script.encode())
            
            print(f"Successfully converted PowerPoint to PDF: {pdf_path}")
            return Path(pdf_path)
        except Exception as e:
            print(f"macOS PowerPoint to PDF conversion error: {str(e)}")
            return None

    def _convert_pptx_linux(self, pptx_path, output_folder, pdf_name):
        """Linux conversion method using LibreOffice"""
        try:
            # Check if LibreOffice is installed
            libreoffice_path = subprocess.check_output(["which", "libreoffice"], universal_newlines=True).strip()
            if not libreoffice_path:
                print("LibreOffice is not installed")
                return None
            
            # Convert directly to PDF using LibreOffice
            pdf_path = os.path.join(output_folder, pdf_name)
            cmd = ["libreoffice", "--headless", "--convert-to", "pdf", 
                   "--outdir", output_folder, pptx_path]
            subprocess.run(cmd, check=True)
            
            # LibreOffice creates PDF with original filename, so we might need to rename
            original_name = os.path.splitext(os.path.basename(pptx_path))[0] + ".pdf"
            original_path = os.path.join(output_folder, original_name)
            
            if original_path != pdf_path and os.path.exists(original_path):
                os.rename(original_path, pdf_path)
            
            print(f"Successfully converted PowerPoint to PDF: {pdf_path}")
            return Path(pdf_path)
        except FileNotFoundError:
            print("LibreOffice is not installed. Install it using:")
            print("  sudo apt-get install libreoffice")
            return None
        except Exception as e:
            print(f"Linux PowerPoint to PDF conversion error: {str(e)}")
            return None

    async def run(self):
        """
        Convert PowerPoint to PDF and then perform OCR on the resulting PDF.
        
        Returns:
            list[dict]: OCR results in the same format as PDFProcessor
        """
        try:
            # Check requirements
            if not self._check_requirements():
                return [{"error": "Required software not installed"}]
            
            # Create temporary folder for PDF
            temp_folder = Path.cwd() / "temp_pptx_pdf"
            temp_folder.mkdir(exist_ok=True)
            
            # Convert PPTX to PDF
            pdf_path = self._convert_pptx_to_pdf(str(temp_folder))
            if pdf_path is None:
                return [{"error": "Failed to convert PowerPoint to PDF"}]
            
            self.temp_pdf_path = pdf_path
            
            # Use PDFProcessor to OCR the generated PDF
            pdf_processor = PDFProcessor(pdf_path)
            ocr_results = await pdf_processor.run()
            
            return ocr_results
            
        except Exception as e:
            print(f"PowerPoint processing error: {e}")
            return [{"error": str(e)}]
        finally:
            # Clean up temporary PDF file
            if self.temp_pdf_path and self.temp_pdf_path.exists():
                try:
                    self.temp_pdf_path.unlink()
                    print(f"Deleted temporary PDF: {self.temp_pdf_path}")
                except Exception as e:
                    print(f"Could not delete temporary PDF: {e}")


# Legacy function compatibility (optional)
# def convert_pptx_to_pdf(pptx_path, output_folder, pdf_name=None):
#     """Legacy function for backward compatibility"""
#     processor = PowerPointProcessor(pptx_path)
#     return processor._convert_pptx_to_pdf(output_folder, pdf_name) is not None
