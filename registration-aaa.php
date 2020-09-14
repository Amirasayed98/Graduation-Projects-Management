<?php
    session_start();
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>registration</title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/registration.css"/>
    
    <script src="js/jquery-3.4.1.min.js"></script>
    <script src="js/bootstrap.min.js"></script>

    
</head>

<body>

  <div class="main">
    <h1>Ready to upload your project !</h1>
    <h3>Make sure to enter a correct and complete data...</h3>
  </div>
   <span id="error">

                     <?php
                     if(isset($_SESSION['errors'])){
                         if(!empty($_SESSION['errors'])){
                             echo '<div class="alert alert-danger">';
                             echo '<ul>';
                             foreach ($_SESSION['errors'] as $error){
                                 echo '<li>' ;
                                    echo $error;
                                 echo '</li>' ;
                             }
                             echo '</ul>';
                             echo   '</div>';
        }
        }
        ?>
        </span>

    <form action="project-registion-data.php" method="post" class="shadow-lg " enctype="multipart/form-data">
      
        <div class="container-fluid require_head">
          <img src="image/logo-main.PNG" class="card-img-top disp" alt="find image">
            <h4 class="right disp">Graduation Project Verifcation Form</h4>
        </div>

        <div class="f2">
        <div class="form-group">
            <label for="TextInput01">Project name <span>*</span></label>
            <input  name="projectname" type="text" id="TextInput01" class="form-control" placeholder="Enter short name*" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['pname'] ?>">
            <div class=" note">*Use 10 or more letters </div>
        </div>
        <div class="form-group">
            <label for="TextInput21">Project description  key words <span>*</span></label>
            <input  name="keywords" type="text" id="TextInput22" class="form-control" placeholder="Enter short key words*" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['keywords'] ?>">
        </div>
        
        <div class="form-row">
            <div class="form-group col-md-6">
              <label for="TextInput02">Project ID <span>*</span></label>
              <input  name="project_id" type="text" class="form-control" id="TextInput02" placeholder="$$$$" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['pid'] ?>">
            </div>
            <div class="form-group col-md-6">
                
                    <label for="validationCustom01">Doctor name <span>*</span></label>
                    <select  name="doctor_name"class="custom-select" id="validationCustom01" required>
                      <option selected disabled value="">Choose...</option>
                      <option value="Hala Zayed" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Hala Zayed') echo 'selected'?>>Hala Zayed</option>
                      <option value="Mazen Selem" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Mazen Selem') echo 'selected'?>>Mazen Selem</option>
                      <option value="Ahmed Taha" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Ahmed Taha') echo 'selected'?>>Ahmed Taha</option>
                      <option value="Mohammed Taha" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Mohammed Taha') echo 'selected'?>>Mohammed Taha</option>
                      <option value="Diaa Salama" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Diaa Salama') echo 'selected'?>>Diaa Salama</option>
                      <option value="Tarek ElSheshtawy" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Tarek ElSheshtawy') echo 'selected'?>>Tarek ElSheshtawy</option>
                      <option value="Mostafa Abd ElSalam" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Mostafa Abd ElSalam') echo 'selected'?>>Mostafa Abd ElSalam</option>
                      <option value="Khaled Foaad" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Khaled Foaad') echo 'selected'?>>Khaled Foaad</option>
                      <option value="Ahmed Hassan" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Ahmed Hassan') echo 'selected'?>>Ahmed Hassan</option>
                      <option value="Metwally Rashad" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Metwally Rashad') echo 'selected'?>>Metwally Rashad</option>
                      <option value="Neven ElSayed"<?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Neven ElSayed') echo 'selected'?>>Neven ElSayed</option>
                      <option value="Fatema Sakr" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Fatema Sakr') echo 'selected'?>>Fatema Sakr</option>
                      <option value="Rasha Orban" <?php if(isset($_SESSION['data']) && $_SESSION['data']['doctor'] == 'Rasha Orban') echo 'selected'?>>Rasha Orban</option>

                    </select>
                    
                  
            </div>
          </div>

        <div class="form-row">
          <div class="col-md-4 mb-3">
            <label for="validationCustom02">Choose Track <span>*</span></label>
            <select  class="custom-select" name="track"  id="validationCustom02" required >
                <option selected disabled value="">Choose...</option>
                <option value="Artficial Intelligence" <?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Artficial Intelligence') echo 'selected'?>>Artficial Intelligence</option>
                <option value="Virtual Reality"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Virtual Reality') echo 'selected'?>>Virtual Reality</option>
                <option value="Machine Learning" <?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Machine Learning') echo 'selected'?>>Machine Learning</option>
                <option value="Image Processing"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Image Processing') echo 'selected'?>>Image Processing</option>
                <option value="Internet Of Things"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Internet Of Things') echo 'selected'?>>Internet Of Things</option>
                <option value="Deep Learning"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Deep Learning') echo 'selected'?>>Deep Learning</option>
                <option value="Web Development"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Web Development') echo 'selected'?>>Web Development</option>
                <option value="Android Development"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Android Development') echo 'selected'?>>Android Development</option>
                <option value="Desktop Application"<?php if(isset($_SESSION['data']) && $_SESSION['data']['ptrack'] == 'Desktop Application') echo 'selected'?>>Desktop Application</option>
            </select>
            
          </div>
          <div class="col-md-4 mb-3">
            <label for="validationCustom03">Department <span>*</span></label>
            <select name="department" class="custom-select" id="validationCustom03" required>
                <option selected disabled value="">Choose...</option>
                <option value="Computer Science" <?php if(isset($_SESSION['data']) && $_SESSION['data']['department'] == 'Computer Science') echo 'selected'?>>Computer Science</option>
                <option value="Information System"  <?php if(isset($_SESSION['data']) && $_SESSION['data']['department'] == 'Information System') echo 'selected'?>>Information System</option>
                <option value="Scientific Computing"  <?php if(isset($_SESSION['data']) && $_SESSION['data']['department'] == 'Scientific Computing') echo 'selected'?>>Scientific Computing</option>


            </select>
            
          </div>
          <div class="col-md-4 mb-3">
            <label for="validationCustom04">Graduation year <span>*</span></label>
            <select name="graduation_year" class="custom-select" id="validationCustom04" required>
                <option selected disabled value="">Choose...</option>
                <option value="2010" <?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2010') echo 'selected'?>>2010</option>
                <option value="2011" <?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2011') echo 'selected'?>>2011</option>
                <option value="2012"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2012') echo 'selected'?>>2012</option>
                <option value="2013"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2013') echo 'selected'?>>2013</option>
                <option value="2014"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2014') echo 'selected'?>>2014</option>
                <option value="2015"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2015') echo 'selected'?>>2015</option>
                <option value="2016"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2016') echo 'selected'?>>2016</option>
                <option value="2017"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2017') echo 'selected'?>>2017</option>
                <option value="2018"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2018') echo 'selected'?>>2018</option>
                <option value="2019"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2019') echo 'selected'?>>2019</option>
                <option value="2020"<?php if(isset($_SESSION['data']) && $_SESSION['data']['gyear'] == '2020') echo 'selected'?>>2020</option>
            </select>
            
          </div>
        </div>
        <div class="mb-3">
            <label for="validationTextarea01">Please write a short and clear description in at least 10 sentences (companies and students will see this description later) <span>*</span></label>
            <textarea name ="description" class="form-control " id="validationTextarea01" placeholder="Project Short Discription*...." required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['description'] ?>"></textarea>
            
          </div>

        <div class="mb-3">
            <h6 class="gray">Upload a project documentation</h6>
             
        </div>
        <div class="custom-file">
            <input   name="uploadfiles" type="file" class="custom-file-input" id="validatedCustomFile" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['uploadfiles'] ?>">
            <!--  <p class="help-block"><span class="label label-info">Note:</span> Please, Select the only files (.pdf) to upload with the size of 100KB only.</p>-->
         
            <label name="documentation" class="custom-file-label" for="validatedCustomFile">Choose file...</label>
            <div class="file note">* File must be a pdf file & his size not exceeds 100 MB </div> 
            
        </div>
        <div class="mb-3">
          <h6 class="gray">Team Data</h6>  
        </div>
        <div class="row">
            <div class="col">
              <input name="leader_name" type="text" class="form-control" placeholder="Team leader name*" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['leadername'] ?>">
            </div>
            <div class="col">
              <input name="leader_id" type="text" class="form-control" placeholder="Team leader National ID*" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['leaderid'] ?>">
            </div>
        </div>
        <div class="row">
          <div class="col email">
            <input  name="leader_email" type="text" class="form-control" placeholder="Team leader E-mail*" required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['leaderemail'] ?>">
          </div>
        </div>

        <div class="mb-3">
            
            <textarea name="members_name" class="form-control email" id="validationTextarea02" placeholder="Team members name*..." required value="<?php if(isset($_SESSION['data'])) echo $_SESSION['data']['membersname'] ?>"></textarea>
            
          </div>
        <div class="form-group">
            <h6>Do you agree that companies see your private data such as (Phone number and E-mail) to contact you if they like your 
                project?
                </h6>
            <div class="check">
              <div class="custom-control custom-radio custom-control-inline">
                    <input type="radio" id="customRadioInline1" name="customRadioInline1" class="custom-control-input">
                    <label class="custom-control-label" for="customRadioInline1">Agree</label>
              </div>
              <div class="custom-control custom-radio custom-control-inline">
                    <input type="radio" id="customRadioInline2" name="customRadioInline1" class="custom-control-input">
                    <label class="custom-control-label" for="customRadioInline2">Don't agree</label>
              </div>
            </div>
          
        </div>
        <button class="btn " type="submit">Submit </button>
    </div>
    </form>

    <script src="bs-custom-file-input-master\dist\bs-custom-file-input.js"></script>

<script>
    
    bsCustomFileInput.init()

   

</script>

</body>
</html>
<?php
if(isset($_SESSION['errors'])){
    unset($_SESSION['errors']);
}
if(isset($_SESSION['data'])){
    unset($_SESSION['data']);
}
?>


